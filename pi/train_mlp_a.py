# Created by shaji at 26/03/2024

import os.path
import torch
import time

from ocatari.core import OCAtari
from tqdm import tqdm
from pi.utils import args_utils
from src import config
from pi import train_utils
from pi.utils.game_utils import create_agent

from pi.utils import game_utils, reason_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils.oc_utils import extract_logic_state_atari
from pi.utils.atari import game_patches
from pi.utils import game_utils, draw_utils


def _action_reassign(action):
    return action


def collect_data_dqn_a(agent, args, buffer_filename, save_buffer):
    oc_name = game_utils.get_ocname(args.m)
    env = OCAtari(oc_name, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    agent.position_norm_factor = obs.shape[0]
    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)

    for game_i in tqdm(range(args.teacher_game_nums), desc=f"Collecting GameBuffer by {agent.agent_type}"):
        env_args.obs, info = env.reset()
        env_args.reset_args(game_i)
        env_args.reset_buffer_game()
        while not env_args.game_over:
            # limit frame rate
            if args.with_explain:
                current_frame_time = time.time()
                if env_args.last_frame_time + env_args.target_frame_duration > current_frame_time:
                    sl = (env_args.last_frame_time + env_args.target_frame_duration) - current_frame_time
                    time.sleep(sl)
                    continue
                env_args.last_frame_time = current_frame_time  # save frame start time for next iteration

            env_args.logic_state, env_args.state_score = extract_logic_state_atari(args, env.objects, args.game_info,
                                                                                   obs.shape[0])
            env_args.past_states.append(env_args.logic_state)
            env_args.obs = env_args.last_obs
            state = env.dqn_obs.to(args.device)
            if env_args.frame_i <= args.jump_frames:
                env_args.action = 0
            else:
                env_args.action, _ = agent(env.dqn_obs.to(env_args.device))
                env_args.action = _action_reassign(env_args.action)
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
            if args.with_explain:
                screen_text = (
                    f"dqn_obj ep: {env_args.game_i}, Rec: {env_args.best_score} \n "
                    f"act: {args.action_names[env_args.action]} re: {env_args.reward}")
                # Red
                env_args.obs[:10, :10] = 0
                env_args.obs[:10, :10, 0] = 255
                # Blue
                env_args.obs[:10, 10:20] = 0
                env_args.obs[:10, 10:20, 2] = 255
                draw_utils.addCustomText(env_args.obs, f"DQN-C",
                                         color=(255, 255, 255), thickness=1, font_size=0.3, pos=[1, 5])
                game_plot = draw_utils.rgb_to_bgr(env_args.obs)
                screen_plot = draw_utils.image_resize(game_plot,
                                                      int(game_plot.shape[0] * env_args.zoom_in),
                                                      int(game_plot.shape[1] * env_args.zoom_in))
                draw_utils.addText(screen_plot, screen_text,
                                   color=(255, 228, 181), thickness=2, font_size=0.6, pos="upper_right")
                video_out = draw_utils.write_video_frame(video_out, screen_plot)
            # update game args
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
    if args.with_explain:
        draw_utils.release_video(video_out)
    if save_buffer:
        game_utils.save_game_buffer(args, env_args, buffer_filename)


def _prepare_mlp_training_data(args, student_agent):
    if args.m == "Pong":
        actions = torch.cat(student_agent.actions, dim=0)[args.stack_num - 1:]
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_pong_kinematics(args, states)
        kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)
        pos_data = [
            kinematic_series_data[:, 1:2],
            kinematic_series_data[:, 2:]
        ]
        args.dqn_a_avg_score = torch.sum(student_agent.buffer_win_rates > 0) / len(student_agent.buffer_win_rates)

    elif args.m == "Asterix":

        actions = torch.cat(student_agent.actions, dim=0)[args.stack_num - 1:]
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_asterix_kinematics(args, states)
        kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)
        pos_data = [
            kinematic_series_data[:, 1:9],
            kinematic_series_data[:, 9:]
        ]
        args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)
    elif args.m == "Boxing":
        actions = torch.cat(student_agent.actions, dim=0)
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_boxing_kinematics(args, states)
        kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)
        action_series_data = train_utils.get_stack_buffer(actions.unsqueeze(1).unsqueeze(1), args.stack_num)
        action_series_data = torch.repeat_interleave(action_series_data, 2, dim=1)

        # kinematic_action_series_data = torch.cat((kinematic_series_data, action_series_data), dim=2)
        actions = actions[args.stack_num - 1:]
        pos_data = [
            kinematic_series_data[:, 1:2]
        ]
        args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)
    elif args.m == "Kangaroo":

        actions = torch.cat(student_agent.actions, dim=0)[args.stack_num - 1:]
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_kangaroo_kinematics(args, states)
        kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)
        pos_data = [
            kinematic_series_data[:, 1:2],
            kinematic_series_data[:, 2:5],
            kinematic_series_data[:, 5:6],
            kinematic_series_data[:, 6:10],
            kinematic_series_data[:, 10:13],
            kinematic_series_data[:, 13:17],
            kinematic_series_data[:, 17:20],
            kinematic_series_data[:, 20:23],
        ]
        args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)
    else:
        raise ValueError
    return pos_data, actions


def train_mlp_a():
    args = args_utils.load_args(config.path_exps, None)


    # Initialize environment
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    num_actions = env.action_space.n

    # learn behaviors from data
    student_agent = create_agent(args, agent_type='smp')
    # collect game buffer from neural agent
    dqn_a_input_shape = env.observation_space.shape
    action_num = len(args.action_names)

    buffer_filename = args.game_buffer_path / f"z_buffer_dqn_a_{args.teacher_game_nums}.json"

    if not os.path.exists(buffer_filename):
        dqn_a_agent = train_utils.load_dqn_a(args, args.model_path)
        dqn_a_agent.agent_type = "DQN-A"
        collect_data_dqn_a(dqn_a_agent, args, buffer_filename, save_buffer=True)

    student_agent.load_atari_buffer(args, buffer_filename)
    pos_data, actions = _prepare_mlp_training_data(args, student_agent)
    # train MLP-A
    obj_type_models = []
    for obj_type in range(len(pos_data)):
        input_tensor = pos_data[obj_type].to(args.device)
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        target_tensor = actions.to(args.device)

        act_pred_model_file = args.trained_model_folder / f"{args.m}_mlp_a_{obj_type}.pth.tar"

        if not os.path.exists(act_pred_model_file):
            action_pred_model = train_utils.train_nn(args, num_actions, input_tensor, target_tensor,
                                                     f"mlp_a_{obj_type}")
            state = {'model': action_pred_model}
            torch.save(state, act_pred_model_file)
        else:
            action_pred_model = torch.load(act_pred_model_file, map_location=torch.device(args.device))["model"]
        obj_type_models.append(action_pred_model)

    return args.dqn_a_avg_score


if __name__ == "__main__":
    dqn_a_avg_score = train_mlp_a()
