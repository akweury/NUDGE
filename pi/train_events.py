# Created by shaji at 05/04/2024
import os.path
import torch
from ocatari.core import OCAtari
from tqdm import tqdm
import time
from pi.utils import args_utils
from src import config
from pi import train_utils
from pi.utils.game_utils import create_agent

from pi.utils import game_utils, reason_utils, draw_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils.oc_utils import extract_logic_state_atari
from pi.utils.atari import game_patches
from pi import train_dqn_t
import random


def collect_data_hla(agent, args, buffer_filename, save_buffer):
    oc_name = game_utils.get_ocname(args.m)
    env = OCAtari(oc_name, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    agent.position_norm_factor = obs.shape[0]
    for game_i in tqdm(range(args.teacher_game_nums), desc=f"Collecting GameBuffer by {agent.agent_type}"):
        env_args.obs, info = env.reset()
        env_args.reset_args(game_i)
        env_args.reset_buffer_game()
        while not env_args.game_over:
            env_args.logic_state, env_args.state_score = extract_logic_state_atari(args, env.objects, args.game_info,
                                                                                   obs.shape[0])
            env_args.past_states.append(env_args.logic_state)
            env_args.obs = env_args.last_obs
            state = env.dqn_obs.to(args.device)
            if env_args.frame_i <= args.jump_frames:
                env_args.action = 0
            else:
                env_args.action, _ = agent(env.dqn_obs.to(env_args.device))
            env_args.obs, env_args.reward, env_args.terminated, env_args.truncated, info = env.step(env_args.action)

            game_patches.atari_frame_patches(args, env_args, info)

            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, agent, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1
                env_args.update_lost_live(args.m, info["lives"])
            else:
                # record game states
                env_args.next_state, env_args.state_score = extract_logic_state_atari(args, env.objects, args.game_info,
                                                                                      obs.shape[0])
                env_args.buffer_frame("dqn_a")
            # update game args
            env_args.update_args()

        if args.m == "Pong":
            if sum(env_args.rewards) > 0:
                env_args.buffer_game(args.zero_reward, args.save_frame)
        elif args.m == "Asterix":
            env_args.buffer_game(args.zero_reward, args.save_frame)
        elif args.m == "Kangaroo":
            env_args.buffer_game(args.zero_reward, args.save_frame)
        elif args.m == "Freeway":
            env_args.buffer_game(args.zero_reward, args.save_frame)
        elif args.m == "Boxing":
            env_args.buffer_game(args.zero_reward, args.save_frame)
        else:
            raise ValueError
        env_args.game_rewards.append(env_args.rewards)

        game_utils.game_over_log(args, agent, env_args)
        env_args.reset_buffer_game()
    env.close()
    game_utils.finish_one_run(env_args, args, agent)
    if save_buffer:
        game_utils.save_game_buffer(args, env_args, buffer_filename)


def _prepare_mlp_training_data(args, states):
    if args.m == "Pong":
        actions = torch.cat(student_agent.actions, dim=0)[args.stack_num - 1:]
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_pong_kinematics(args, states)
        kinematic_data_new = reason_utils.extract_pong_kinematics_new(args, states)

        args.dqn_a_avg_score = torch.sum(student_agent.buffer_win_rates > 0) / len(student_agent.buffer_win_rates)

    elif args.m == "Asterix":

        actions = torch.cat(student_agent.actions, dim=0)[args.stack_num - 1:]
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_asterix_kinematics(args, states)
        kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)

        args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)
    elif args.m == "Boxing":
        actions = torch.cat(student_agent.actions, dim=0)[args.stack_num - 1:]
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_boxing_kinematics(args, states)
        kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)

        args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)
    elif args.m == "Freeway":
        actions = torch.cat(student_agent.actions, dim=0)
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_freeway_kinematics(args, states)
        pos_data = [
            kinematic_data[:, 1:2],
            kinematic_data[:, 2:]
        ]
        args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)

    elif args.m == "Kangaroo":
        kinematic_data = reason_utils.extract_kangaroo_kinematics(args, states)
    else:
        raise ValueError
    return kinematic_data


def get_target_hla(last_target):
    if last_target == "Ladder":
        hla = "goto"
    elif last_target == "Monkey":
        hla = "kill"
    elif last_target == "Fruit":
        hla = "goto"
    elif last_target == "FallingCoconut":
        hla = "avoid"
    elif last_target == "ThrownCoconut":
        hla = "avoid"
    else:
        raise ValueError
    return hla


def reasoning_action(target_kinematic, hla_target):
    dist = target_kinematic[[2, 3]]
    action = None
    if hla_target == "avoid":
        pass
    elif hla_target == "kill":
        pass
    elif hla_target == "goto":
        pass
    else:
        raise ValueError

    return action


from pi import train_transformer


def _reason_action(args, agent, env, env_args):
    action = None
    if args.last_target is None:
        args.last_target = "Player"

    state_kinematic = reason_utils.extract_kangaroo_kinematics(args, env_args.past_states)
    target_indices = args.same_others[args.row_names.index(args.last_target)]
    target_kinematic = state_kinematic[-1, target_indices]
    min_dist = 0.05
    target_dist = target_kinematic[:, [2, 3]].abs().sum(dim=1)
    closest_target = target_dist.argmin()
    if target_dist.min() < min_dist:
        # check if target satisfied
        args.last_target = train_transformer.eval_transformer([[args.last_target]],
                                                              args.transformer_dataset, args.transformer_file)
        target_indices = args.same_others[args.row_names.index(args.last_target)]
        target_kinematic = state_kinematic[-1, target_indices]
        target_dist = target_kinematic[:, [2, 3]].abs().sum(dim=1)
        closest_target = target_dist.argmin()

    # goto the closest target
    hla_target = get_target_hla(args.last_target)
    reasoning_action(target_kinematic[closest_target], hla_target)

    return action


def play_game(args, agent, buffer_filename):
    oc_name = game_utils.get_ocname(args.m)
    # load mlp_a
    obj_type_num = len(args.game_info["obj_info"]) - 1

    env = OCAtari(oc_name, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    agent.position_norm_factor = obs.shape[0]

    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)
    args.last_target = None
    for game_i in tqdm(range(args.dqn_c_episode_num), desc=f"Collecting GameBuffer by {agent.agent_type}"):
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

            # predict object id
            env_args.logic_state, _ = extract_logic_state_atari(args, env.objects, args.game_info, obs.shape[0])
            env_args.past_states.append(env_args.logic_state)

            if env_args.frame_i <= args.jump_frames:
                action = torch.tensor([[0]]).to(args.device)
                obj_pred = torch.tensor([[0]]).to(args.device)
            else:
                action = _reason_action(args, agent, env, env_args)

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
                env_args.target = obj_pred.item()
                env_args.obj_pred = obj_pred.reshape(-1).item()
                env_args.buffer_frame("dqn_t")
            env_args.frame_i += 1
            if args.with_explain:
                screen_text = (
                    f"dqn_obj ep: {env_args.game_i}, Rec: {env_args.best_score} \n "
                    f"obj: {args.row_names[obj_pred + 1]}, act: {args.action_names[action]} re: {env_args.reward}")
                # Red
                env_args.obs[:10, :10] = 0
                env_args.obs[:10, :10, 0] = 255
                # Blue
                env_args.obs[:10, 10:20] = 0
                env_args.obs[:10, 10:20, 2] = 255
                draw_utils.addCustomText(env_args.obs, f"dqn_t",
                                         color=(255, 255, 255), thickness=1, font_size=0.3, pos=[1, 5])
                game_plot = draw_utils.rgb_to_bgr(env_args.obs)
                screen_plot = draw_utils.image_resize(game_plot,
                                                      int(game_plot.shape[0] * env_args.zoom_in),
                                                      int(game_plot.shape[1] * env_args.zoom_in))
                draw_utils.addText(screen_plot, screen_text,
                                   color=(255, 228, 181), thickness=2, font_size=0.6, pos="upper_right")
                video_out = draw_utils.write_video_frame(video_out, screen_plot)

            env_args.reward = torch.tensor(env_args.reward).reshape(1).to(args.device)
            next_state = env.dqn_obs.to(args.device) if not env_args.terminated else None

        if args.m == "Pong":
            if sum(env_args.rewards) > 0:
                env_args.buffer_game(args.zero_reward, args.save_frame)
                game_utils.game_over_log(args, agent, env_args)
        elif args.m == "Asterix":
            env_args.buffer_game(args.zero_reward, args.save_frame)
            game_utils.game_over_log(args, agent, env_args)
        elif args.m == "Kangaroo":
            env_args.buffer_game(args.zero_reward, args.save_frame)
            game_utils.game_over_log(args, agent, env_args)
        else:
            raise ValueError

        env_args.reset_buffer_game()

    env.close()
    game_utils.finish_one_run(env_args, args, agent)
    if args.with_explain:
        draw_utils.release_video(video_out)
    game_utils.save_game_buffer(args, env_args, buffer_filename)


def _get_events(args, kinematic_data, state_actions):
    event_obj_ids = []
    event_obj_names = []
    event_dists = []
    event_positions = []
    event_actions = []
    min_dist = 0.05
    for s_i in range(kinematic_data.shape[0]):
        if s_i == 0:
            event_obj_ids.append(0)
            event_obj_names.append(args.row_names[0])
            event_dists.append(kinematic_data[s_i, 0, [2, 3]])
            event_actions.append(state_actions[s_i])
            event_positions.append(kinematic_data[s_i, 0, [0, 1]])
            continue
        rel_dist = kinematic_data[s_i, 1:, [2, 3]]
        closest_dist = rel_dist.abs().sum(dim=1).min()
        closest_obj = rel_dist.abs().sum(dim=1).argmin() + 1
        if closest_dist < min_dist:
            event_obj_ids.append(closest_obj)
            event_obj_names.append(args.row_names[closest_obj])
            event_dists.append(rel_dist[closest_obj - 1])
            event_actions.append(state_actions[s_i])
            event_positions.append(kinematic_data[s_i, 0, [0, 1]])

    group_dists = []
    group_names = []
    group_positions = []
    group_ids = []
    group_actions = []
    last_obj_id = -1
    dists = []
    actions = []
    positions = []
    # merge events
    for e_i in range(len(event_obj_ids)):
        current_obj_id = event_obj_ids[e_i]
        if current_obj_id != last_obj_id and e_i != 0:
            group_dists.append(dists)
            group_positions.append(torch.cat(positions, dim=0).mean(dim=0).unsqueeze(0))
            group_actions.append(actions)
            group_names.append(args.row_names[last_obj_id])
            group_ids.append(last_obj_id)
            actions = []
            dists = []
            positions = []
        dists.append(event_dists[e_i])
        positions.append(event_positions[e_i].unsqueeze(0))
        actions.append(args.action_names[state_actions[e_i]])
        last_obj_id = current_obj_id
        if e_i == len(event_obj_ids) - 1:
            group_dists.append(dists)
            group_positions.append(torch.cat(positions, dim=0).mean(dim=0).unsqueeze(0))
            group_names.append(args.row_names[last_obj_id])
            group_ids.append(last_obj_id)
            group_actions.append(actions)
    group_positions = torch.cat(group_positions, dim=0)
    return group_ids, group_names, group_dists, group_positions, group_actions


def train_next_obj_predictor():
    # game buffer
    args = args_utils.load_args(config.path_exps, None)
    # train mlp-hla
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    num_actions = env.action_space.n
    dqn_t_input_shape = env.observation_space.shape
    obj_type_num = args.game_info["state_row_num"] - 1
    student_agent = create_agent(args, agent_type='smp')
    # collect game buffer from neural agent
    buffer_filename = args.game_buffer_path / f"z_buffer_dqn_a_{args.teacher_game_nums}.json"
    student_agent.load_atari_buffer_by_games(args, buffer_filename)

    game_names = []
    game_dists = []
    game_positions = []
    game_ids = []
    for game_i in range(len(student_agent.states)):
        kinematic_data = _prepare_mlp_training_data(args, student_agent.states[game_i])
        actions = student_agent.buffer_actions[game_i]
        group_ids, group_names, group_dists, group_positions, group_actions = _get_events(args, kinematic_data, actions)
        game_positions.append(group_positions)
        game_names.append(group_names)
        game_dists.append(group_dists)
        game_ids.append(group_ids)
    from pi import train_transformer
    from sklearn.model_selection import train_test_split
    target_transformer_file = args.output_folder / "target_transformer_model.pth"
    transformer_dataset = train_transformer.WordSequenceDataset(game_names)
    train_transformer.train_transformer(transformer_dataset, target_transformer_file)

    position_transformer_file = args.output_folder / "position_transformer_model.pth"
    # position transformer
    positions = train_transformer.position_dataset(game_positions)
    train_data, val_data = train_test_split(positions, test_size=0.2)
    train_transformer.train_position_transformer(train_data, val_data, position_transformer_file)

    return args, student_agent, buffer_filename


if __name__ == "__main__":
    args, student_agent, buffer_filename = train_next_obj_predictor()
    play_game(args, student_agent, buffer_filename)
