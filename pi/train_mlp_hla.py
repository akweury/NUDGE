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


def _prepare_mlp_training_data(args, student_agent):
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

        actions = torch.cat(student_agent.actions, dim=0)[args.stack_num - 1:]
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_kangaroo_kinematics(args, states)
        kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)

        args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)
    else:
        raise ValueError
    return kinematic_data, kinematic_data_new, actions


def collect_data_dqn_t(agent, args, buffer_filename, save_buffer):
    oc_name = game_utils.get_ocname(args.m)
    # load mlp_a
    obj_type_num = len(args.game_info["obj_info"]) - 1
    mlp_a = train_utils.load_mlp_a(args, args.trained_model_folder, obj_type_num, args.m)
    # load MLP-C
    mlp_c = train_utils.load_mlp_c(args)

    env = OCAtari(oc_name, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    agent.position_norm_factor = obs.shape[0]

    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)

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
                action, obj_pred = train_dqn_t._reason_action(args, agent, env, env_args, mlp_a, mlp_c)

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
            # Store the transition in memory
            agent.memory.push(state, obj_pred, next_state, env_args.reward, env_args.terminated)

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
    if save_buffer:
        game_utils.save_game_buffer(args, env_args, buffer_filename)


def _action2hla(actions, c_ids, kinematic_data):
    hla = []
    hla_input = []
    # 0 noop
    # 1 fire
    # 2 align
    # 3 away
    action_vectors = reason_utils.action_to_vector_pong(actions).to(actions.device)

    for d_i in tqdm(range(100000)):
        action = actions[d_i]
        action_vector = action_vectors[d_i]

        data = kinematic_data[d_i, c_ids[d_i] + 1]
        do_intersect = reason_utils.target_to_vector_pong(data, action_vector)
        # detect collision and then predict if they are aligning or not
        hla_input.append(data[2:].unsqueeze(0))

        # HLA: noop
        if action in [0]:
            hla.append(0)
        # HLA: fire
        elif action in [1]:
            hla.append(1)
        # check behavior

        elif do_intersect:
            hla.append(2)
        else:
            hla.append(3)
    hla = torch.tensor(hla).to(actions.device)
    hla_input = torch.cat(hla_input, dim=0).to(actions.device)
    return hla, hla_input


def train_mlp_hla():
    # game buffer
    args = args_utils.load_args(config.path_exps, None)
    if args.m not in ["Pong"]:
        return

    # train mlp-hla
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    num_actions = env.action_space.n
    dqn_t_input_shape = env.observation_space.shape
    obj_type_num = args.game_info["state_row_num"] - 1
    student_agent = create_agent(args, agent_type='smp')
    # collect game buffer from neural agent
    buffer_filename = args.game_buffer_path / f"z_buffer_dqn_a_{args.teacher_game_nums}.json"
    student_agent.load_atari_buffer(args, buffer_filename)
    kinematic_data, kinematic_data_new, actions = _prepare_mlp_training_data(args, student_agent)

    # load MLP-C
    mlp_c = train_utils.load_mlp_c(args)
    # predict collective
    kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)
    input_c_tensor = kinematic_series_data[:, 1:].view(kinematic_series_data.shape[0], -1)
    collective_id_mlp_conf = mlp_c(input_c_tensor)
    collective_id_mlp = collective_id_mlp_conf.argmax(dim=1)

    # convert action to high level action
    hla, hla_input = _action2hla(actions, collective_id_mlp, kinematic_data_new[args.stack_num - 1:])
    hla_num = 4
    hla_model = args.trained_model_folder / f"{args.m}_mlp_hla.pth.tar"

    if not os.path.exists(hla_model):
        target_pred_model = train_utils.train_nn(args, hla_num, hla_input, hla, f"mlp_hla")
        state = {'model': target_pred_model}
        torch.save(state, hla_model)
    else:
        target_pred_model = torch.load(hla_model)["model"]

    return args.dqn_a_avg_score


if __name__ == "__main__":
    train_mlp_hla()
