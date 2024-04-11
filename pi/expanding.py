# Created by jing at 10.04.24

import os.path
import torch
import time
from rtpt import RTPT

from ocatari.core import OCAtari
from tqdm import tqdm
from pi.utils import args_utils
from src import config
from pi import train_utils
from pi.utils.game_utils import create_agent

from pi.utils import game_utils, reason_utils, math_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils.oc_utils import extract_logic_state_atari
from pi.utils.atari import game_patches
from pi.utils import game_utils, draw_utils, file_utils


def collect_data(args, agent):
    buffer_filename = args.game_buffer_path / f"z_buffer_dqn_a_{args.teacher_game_nums}.json"
    if not os.path.exists(buffer_filename):
        game_utils.collect_data_dqn_a(agent, args, buffer_filename, save_buffer=True)

    game_buffer = game_utils.load_atari_buffer(args, buffer_filename)
    return game_buffer


def _get_kinematic_states(args, game_buffer):
    if args.m == "Freeway":
        actions = torch.cat(game_buffer["actions"], dim=0)
        states = torch.cat(game_buffer["states"], dim=0)
        kinematic_data = reason_utils.extract_freeway_kinematics(args, states)
    else:
        raise ValueError
    return kinematic_data, actions


def in_range(kinematic_data, new_param_range):
    param_num = kinematic_data.shape[1]
    data_range = torch.repeat_interleave(new_param_range.unsqueeze(0), kinematic_data.shape[0], dim=0)
    min_range_satisfaction = torch.sum((kinematic_data - data_range[:, :, 0]) >= 0, dim=1) == param_num
    max_range_satisfaction = torch.sum((data_range[:, :, 1] - kinematic_data) >= 0, dim=1) == param_num
    mask_in_range = max_range_satisfaction & min_range_satisfaction
    return mask_in_range


def check_range(parameter_state, pred_ranges):
    param_num = parameter_state.shape[1]
    data_state = torch.repeat_interleave(parameter_state, pred_ranges.shape[0], dim=0)
    min_range_satisfaction = torch.sum((data_state - pred_ranges[:, :, 0]) >= 0, dim=1) == param_num
    max_range_satisfaction = torch.sum((pred_ranges[:, :, 1] - data_state) >= 0, dim=1) == param_num
    mask_in_range = max_range_satisfaction & min_range_satisfaction
    in_range_preds = pred_ranges[mask_in_range]
    return mask_in_range


def same_action(actions, label_action):
    mask_action = actions == label_action
    return mask_action


def _pi_expanding(frame_i, parameter_states, actions):
    # expand

    frame_data = parameter_states[frame_i]
    param_range = torch.repeat_interleave(frame_data.unsqueeze(1), 2, dim=1)

    param_deltas = torch.tensor([[0, 0.01], [-0.01, 0]]).to(parameter_states.device)
    mask_same_action = same_action(actions, actions[frame_i])

    scores = []
    num_cover_frames_best = 0
    for param_delta in param_deltas:
        for p_i in range(2, len(frame_data)):
            # update param range
            score_p_i_best = 0
            param_step = torch.zeros_like(param_range).to(parameter_states.device)
            while True:
                param_step[p_i] = param_delta
                if param_step.abs().sum() > 0.01:
                    print("")
                new_param_range = param_range + param_step
                new_param_range[new_param_range < -1] = -1
                new_param_range[new_param_range > 1] = 1
                if torch.equal(new_param_range, param_range):
                    break
                # evaluate new range and acquire score
                mask_in_range = in_range(parameter_states, new_param_range)
                num_cover_frames = (mask_in_range & mask_same_action).sum()
                score_new_range = num_cover_frames / torch.sum(mask_in_range)
                if score_new_range == 1:
                    score_p_i_best = score_new_range
                    param_range = new_param_range
                    if num_cover_frames > num_cover_frames_best:
                        num_cover_frames_best = num_cover_frames
                else:
                    break
            scores.append(score_p_i_best)
    inv_score = torch.tensor(scores).max()
    return param_range, inv_score, num_cover_frames_best


def _pi_elimination(frame_i, parameter_states, actions, inv_pred, score):
    num_cover_frames = 0
    mask_param = torch.ones(inv_pred.shape[0], dtype=torch.bool)
    mask_same_action = same_action(actions, actions[frame_i])

    for p_i in range(2, inv_pred.shape[0]):
        mask_delta = torch.ones(inv_pred.shape[0], dtype=torch.bool)
        mask_delta[p_i] = False
        mask_param_new = mask_param & mask_delta

        mask_in_range = in_range(parameter_states[:, mask_param_new], inv_pred[mask_param_new])
        num_cover_frames = (mask_in_range & mask_same_action).sum()
        score_new = num_cover_frames / torch.sum(mask_in_range)
        if score_new >= score:
            mask_param = mask_param_new
            score = score_new
            if inv_pred[p_i][0] != -1 and inv_pred[p_i][1] != 1:
                print("")

    return mask_param, score, num_cover_frames


def _get_parameter_states(kinematic_data):
    player_pos = kinematic_data[:, 0:1, :2].reshape(kinematic_data.shape[0], -1)
    others_state = kinematic_data[:, 1:, 2:].reshape(kinematic_data.shape[0], -1)
    parameter_state = torch.cat((player_pos, others_state), dim=1)

    parameter_state = math_utils.closest_one_percent(parameter_state, 0.01)
    return parameter_state


def range_to_text(prefix, range_tensor):
    text = ""
    if range_tensor[0, 0] > -1:
        text += f"{prefix}_>{range_tensor[0, 0]:.2f}_"
    if range_tensor[0, 1] < 1:
        text += f"{prefix}_<{range_tensor[0, 1]:.2f}_"
    return text


def _pi_text(args, inv_pred):
    pos = (inv_pred[0, 0], inv_pred[1, 0])
    text = f"player_at({pos[0].item():.2f},{pos[1].item():.2f})_"
    trivial_pred = True
    for obj_i in range(1, len(args.row_names)):
        obj_name = f"{args.row_names[obj_i]}{obj_i}_"
        obj_index = 2 + (obj_i - 1) * 4
        x_pos = range_to_text("x", inv_pred[obj_index:obj_index + 1])
        y_pos = range_to_text("y", inv_pred[obj_index + 1:obj_index + 2])
        x_velo = range_to_text("vx", inv_pred[obj_index + 2:obj_index + 3])
        y_velo = range_to_text("vy", inv_pred[obj_index + 3:obj_index + 4])
        kinematic_text = f"{x_pos}{y_pos}{x_velo}{y_velo}"
        obj_text = ""
        if kinematic_text != "":
            obj_text = f"{obj_name}({x_pos}{y_pos}{x_velo}{y_velo})_"
            trivial_pred = False
        text += obj_text
    return text, trivial_pred


def save_pred(frame_i, inv_pred, action, inv_pred_file):
    data = torch.load(inv_pred_file)
    data["inv_pred"].append(inv_pred.tolist())
    data["inv_pred_frame_i"].append(frame_i)
    data["action"].append(action.tolist())
    torch.save(data, inv_pred_file)


def create_inv_file(file_name):
    data = {"inv_pred": [], "inv_pred_frame_i": [], "action": []}
    torch.save(data, file_name)


def init_pred_file(file_name, start_frame):
    if os.path.exists(file_name):
        frame_i = torch.load(file_name)["inv_pred_frame_i"]
        if len(frame_i) > 0:
            start_frame = frame_i[-1]
    else:
        create_inv_file(file_name)
    return start_frame


def remove_redundancy(parameter_states, actions):
    unique_states = parameter_states.unique(dim=0)
    unique_states_actions = []
    for s_i in tqdm(range(len(unique_states)), desc="Removing redundancies"):
        mask_same_states = (parameter_states - unique_states[s_i]).sum(dim=1) == 0
        actions_state = actions[mask_same_states]
        # select one action
        unique_values, counts = actions_state.unique(return_counts=True)
        max_count_index = counts.argmax()
        most_frequent_action = unique_values[max_count_index]
        unique_states_actions.append(most_frequent_action.view(1))
    unique_states_actions = torch.cat(unique_states_actions, dim=0)
    return unique_states, unique_states_actions


def main():
    args = args_utils.load_args(config.path_exps, None)

    rtpt = RTPT(name_initials='JS', experiment_name=f"{args.m}_{args.start_frame}_{args.end_frame}",
                max_iterations=args.end_frame - args.start_frame)
    # Start the RTPT tracking
    rtpt.start()

    if args.test:
        test()
    else:
        # Initialize environment
        env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
        obs, info = env.reset()
        args.num_actions = env.action_space.n
        dqn_a_input_shape = env.observation_space.shape
        from pi.utils import file_utils
        args.log_file = file_utils.create_log_file(args.trained_model_folder, "pi")
        # learn behaviors from data
        teacher_agent = create_agent(args, agent_type=args.teacher_agent)
        # collect game buffer from neural agent
        game_buffer = collect_data(args, teacher_agent)
        kinematic_data, actions = _get_kinematic_states(args, game_buffer)
        kinematic_data = kinematic_data.to(args.device)
        actions = actions.to(args.device)
        parameter_states = _get_parameter_states(kinematic_data)
        parameter_states, actions = remove_redundancy(parameter_states, actions)
        # train the model
        file_name = args.trained_model_folder / f"inv_pred_{args.start_frame}_{args.end_frame}.pth"
        start_frame_i = init_pred_file(file_name, args.start_frame)
        inv_preds = []
        inv_pred_scores = []
        end_frame = min(args.end_frame, len(parameter_states))
        for frame_i in tqdm(range(start_frame_i, end_frame), desc=f"Frame"):
            inv_pred, score, num_cover_frames = _pi_expanding(frame_i, parameter_states, actions)
            # mask_param, score, num_cover_frames = _pi_elimination(frame_i, parameter_states, actions, inv_pred, score)
            text, trivial_pred = _pi_text(args, inv_pred)

            if not trivial_pred:
                file_utils.add_lines(
                    f"(inv {frame_i}) {num_cover_frames} {args.action_names[actions[frame_i]]}:-{text}",
                    args.log_file)
                inv_preds.append(inv_pred)
                inv_pred_scores.append(score)
                save_pred(frame_i, inv_pred, actions[frame_i], file_name)


def init_env(args):
    # Initialize environment
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    args.num_actions = env.action_space.n
    args.log_file = file_utils.create_log_file(args.trained_model_folder, "pi")
    return env, obs, info


def _reason_action(args, env_args, inv_preds, inv_pred_actions):
    state = torch.tensor(env_args.logic_state).unsqueeze(0).to(args.device)
    kinematic_data = reason_utils.extract_freeway_kinematics(args, state).to(args.device)
    parameter_states = _get_parameter_states(kinematic_data)
    mask_in_range = check_range(parameter_states, inv_preds)
    action_in_range = inv_pred_actions[mask_in_range].unique()
    if len(action_in_range) != 1:
        print("")
    action = action_in_range[0]
    return action


def play_with_pi(args, inv_preds, inv_pred_actions):
    env, obs, info = init_env(args)
    agent = create_agent(args, agent_type="smp")
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)
    for game_i in tqdm(range(args.teacher_game_nums), desc=f"Collecting GameBuffer by {agent.agent_type}"):
        env_args.obs, info = env.reset()
        env_args.reset_args(game_i)
        env_args.reset_buffer_game()
        while not env_args.game_over:
            if args.with_explain:
                current_frame_time = time.time()
                if env_args.last_frame_time + env_args.target_frame_duration > current_frame_time:
                    sl = (env_args.last_frame_time + env_args.target_frame_duration) - current_frame_time
                    time.sleep(sl)
                    continue
                env_args.last_frame_time = current_frame_time  # save frame start time for next iteration

            env_args.logic_state, _ = extract_logic_state_atari(args, env.objects, args.game_info, obs.shape[0])
            env_args.past_states.append(env_args.logic_state)

            if env_args.frame_i <= args.jump_frames:
                action = torch.tensor([[0]]).to(args.device)
            else:
                action = _reason_action(args, env_args, inv_preds, inv_pred_actions)

            state = env.dqn_obs.to(args.device)
            env_args.obs, env_args.reward, env_args.terminated, env_args.truncated, info = env.step(action)
            game_patches.atari_frame_patches(args, env_args, info)
            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, agent, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1
                env_args.update_lost_live(args.m, info["lives"])
            else:
                # record game states
                env_args.next_state, env_args.state_score = extract_logic_state_atari(args, env.objects, args.game_info,
                                                                                      obs.shape[0])

            env_args.frame_i += 1
            if args.with_explain:
                screen_text = (
                    f"dqn_obj ep: {env_args.game_i}, Rec: {env_args.best_score} \n "
                    f"act: {args.action_names[action]} re: {env_args.reward}")
                # Red
                env_args.obs[:10, :10] = 0
                env_args.obs[:10, :10, 0] = 255
                # Blue
                env_args.obs[:10, 10:20] = 0
                env_args.obs[:10, 10:20, 2] = 255
                draw_utils.addCustomText(env_args.obs, f"dqn_obj",
                                         color=(255, 255, 255), thickness=1, font_size=0.3, pos=[1, 5])
                game_plot = draw_utils.rgb_to_bgr(env_args.obs)
                screen_plot = draw_utils.image_resize(game_plot,
                                                      int(game_plot.shape[0] * env_args.zoom_in),
                                                      int(game_plot.shape[1] * env_args.zoom_in))
                draw_utils.addText(screen_plot, screen_text,
                                   color=(255, 228, 181), thickness=2, font_size=0.6, pos="upper_right")
                video_out = draw_utils.write_video_frame(video_out, screen_plot)

            env_args.reward = torch.tensor(env_args.reward).reshape(1).to(args.device)

        game_utils.game_over_log(args, agent, env_args)

        env_args.reset_buffer_game()

    env.close()
    game_utils.finish_one_run(env_args, args, agent)
    if args.with_explain:
        draw_utils.release_video(video_out)


def collect_pi(args):
    files = file_utils.all_file_in_folder(args.trained_model_folder)
    files = [file for file in files if "inv_pred_" in file]
    inv_preds = []
    actions = []
    for file in files:
        data = torch.load(file, map_location=torch.device(args.device))
        action = torch.tensor(data["action"]).to(args.device)
        inv_pred = torch.tensor(data["inv_pred"]).to(args.device)
        inv_pred = [pred.unsqueeze(0) for pred in inv_pred]
        inv_pred = torch.cat(inv_pred, dim=0)
        actions.append(action)
        inv_preds.append(inv_pred)
    inv_preds = torch.cat(inv_preds, dim=0)
    actions = torch.cat(actions, dim=0)

    return inv_preds, actions


def test():
    args = args_utils.load_args(config.path_exps, None)
    # collect game buffer from neural agent
    inv_preds, actions = collect_pi(args)
    # play the game
    play_with_pi(args, inv_preds, actions)


if __name__ == "__main__":
    main()
