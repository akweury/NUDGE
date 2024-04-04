# Created by shaji at 27/03/2024


from tqdm import tqdm
import os
import time
import torch
from ocatari.core import OCAtari

from pi.utils import game_utils, draw_utils, math_utils, reason_utils, file_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils import args_utils
from src import config
from pi.utils.atari import game_patches
from pi.utils.oc_utils import extract_logic_state_atari
from pi import train_utils, play


def _reason_action(args, agent, env, env_args, mlp_a, mlp_c):
    behavior_id = agent.select_action(env.dqn_obs.to(env_args.device))
    # determine relation related objects
    if args.m == "Asterix":
        state_kinematic = reason_utils.extract_asterix_kinematics(args, env_args.past_states)
        collective_indices, collective_id_dqn = reason_utils.asterix_obj_to_collective(behavior_id)
    elif args.m == "Pong":
        state_kinematic = reason_utils.extract_pong_kinematics(args, env_args.past_states)
    elif args.m == "Kangaroo":
        state_kinematic = reason_utils.extract_kangaroo_kinematics(args, env_args.past_states)
        collective_indices, collective_id_dqn = reason_utils.kangaroo_obj_to_collective(behavior_id + 1)
    else:
        raise ValueError

    # determin object types
    kinematic_series_data = train_utils.get_stack_buffer(state_kinematic, args.stack_num)
    input_c_tensor = kinematic_series_data[-1, 1:].reshape(1, -1)

    collective_id_mlp_conf = mlp_c(input_c_tensor)
    collective_id_mlp = collective_id_mlp_conf.argmax()

    action, beh_text = reason_utils.get_behavior_action(behavior_id, state_kinematic[-1, collective_id_mlp + 1])
    return action, behavior_id, beh_text, collective_id_mlp


def train_dqn_rule():
    BATCH_SIZE = 32
    MEMORY_SIZE = 1000000
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    LEARNING_RATE = 0.00025
    TARGET_UPDATE_FREQ = 5
    EPISODES = 1000

    args = args_utils.load_args(config.path_exps, None)
    if args.m not in ["Pong"]:
        return

    # load MLP-A
    obj_type_num = len(args.game_info["obj_info"]) - 1
    mlp_a = train_utils.load_mlp_a(args, args.trained_model_folder, obj_type_num, args.m)

    # load MLP-C
    mlp_c = train_utils.load_mlp_c(args)

    # Initialize environment
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    input_shape = env.observation_space.shape

    # Initialize agent
    num_behaviors = 6
    agent = train_utils.DQNAgent(args, input_shape, num_behaviors)
    agent.agent_type = "DQN-R"
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    agent.learn_performance = []
    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)

    if args.resume:
        files = os.listdir(args.trained_model_folder)
        dqn_model_files = [file for file in files if f'dqn_rule' in file and ".pth" in file]
        if len(dqn_model_files) == 0:
            start_game_i = 0
        else:
            dqn_model_file = dqn_model_files[0]
            start_game_i = int(dqn_model_file.split("dqn_rule_")[1].split(".")[0]) + 1
            file_dict = torch.load(args.trained_model_folder / dqn_model_file, map_location=torch.device(args.device))
            state_dict = file_dict["state_dict"]
            agent.learn_performance = file_dict["learn_performance"]

            agent.policy_net.load_state_dict(state_dict)
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            agent.target_net.eval()
    else:
        start_game_i = 0

    for game_i in tqdm(range(start_game_i, args.dqn_t_episode_num), desc=f"Training agent  {agent.agent_type}"):
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

            logic_state, _ = extract_logic_state_atari(args, env.objects, args.game_info, obs.shape[0])
            env_args.past_states.append(logic_state)
            if env_args.frame_i <= args.jump_frames:
                action = torch.tensor([[0]]).to(args.device)
                behavior_id = torch.tensor([[0]]).to(args.device)
                beh_text = "noop"
                c_id = -1
            else:
                action, behavior_id, beh_text, c_id = _reason_action(args, agent, env, env_args, mlp_a, mlp_c)

            state = env.dqn_obs.to(args.device)
            env_args.obs, env_args.reward, env_args.terminated, env_args.truncated, info = env.step(action)
            game_patches.atari_frame_patches(args, env_args, info)
            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, agent, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1
                env_args.update_lost_live(args.m, info["lives"])
            if args.with_explain:
                try:
                    screen_text = (
                        f"DQN_R ep: {env_args.game_i}, Rec: {env_args.best_score} \n "
                        f"B: {beh_text}, T: {args.game_info['obj_info'][c_id + 1][0]} A: {args.action_names[action]} re: {env_args.reward}")
                except IndexError:
                    print("")
                # Red
                env_args.obs[:10, :10] = 0
                env_args.obs[:10, :10, 0] = 255
                # Blue
                env_args.obs[:10, 10:20] = 0
                env_args.obs[:10, 10:20, 2] = 255
                draw_utils.addCustomText(env_args.obs, f"DQN-T",
                                         color=(255, 255, 255), thickness=1, font_size=0.3, pos=[1, 5])
                game_plot = draw_utils.rgb_to_bgr(env_args.obs)
                screen_plot = draw_utils.image_resize(game_plot,
                                                      int(game_plot.shape[0] * env_args.zoom_in),
                                                      int(game_plot.shape[1] * env_args.zoom_in))
                draw_utils.addText(screen_plot, screen_text,
                                   color=(255, 228, 181), thickness=2, font_size=0.6, pos="upper_right")
                video_out = draw_utils.write_video_frame(video_out, screen_plot)
            # update game args
            env_args.update_args()
            env_args.rewards.append(env_args.reward)
            env_args.reward = torch.tensor(env_args.reward).reshape(1).to(args.device)
            next_state = env.dqn_obs.to(args.device) if not env_args.terminated else None
            if env_args.frame_i <= args.jump_frames:
                # Store the transition in memory
                agent.memory.push(state, behavior_id, next_state, env_args.reward, env_args.terminated)
                # Perform one step of optimization (on the target network)
                agent.optimize_model()
        # Update the target network, copying all weights and biases in DQN
        if game_i % TARGET_UPDATE_FREQ == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Decay epsilon
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(EPSILON_MIN, EPSILON)

        # env_args.buffer_game(args.zero_reward, args.save_frame)
        env_args.win_rate[game_i] = sum(env_args.rewards[:-1])  # update ep score
        env_args.reset_buffer_game()
        if game_i > args.print_freq and game_i % args.print_freq == 1:
            agent.learn_performance.append(sum(env_args.win_rate[game_i - args.print_freq:game_i]))

            line_chart_data = torch.tensor(agent.learn_performance)
            draw_utils.plot_line_chart(line_chart_data.unsqueeze(0), path=args.trained_model_folder,
                                       labels=[f"total_score_every_{args.print_freq}"],
                                       title=f"{args.m}_learn_dqn_rule_sum_past_{args.print_freq}",
                                       figure_size=(10, 10))
            # save model
            last_epoch_save_path = args.trained_model_folder / f'dqn_r_{game_i + 1 - args.print_freq}.pth'
            save_path = args.trained_model_folder / f'dqn_r_{game_i + 1}.pth'
            if os.path.exists(last_epoch_save_path):
                os.remove(last_epoch_save_path)
            from pi.utils import file_utils
            file_utils.save_agent(save_path, agent, env_args)

    env.close()
    game_utils.finish_one_run(env_args, args, agent)
    buffer_filename = args.game_buffer_path / f"learn_buffer_dqn_r_{args.teacher_game_nums}.json"
    game_utils.save_game_buffer(args, env_args, buffer_filename)
    if args.with_explain:
        draw_utils.release_video(video_out)


if __name__ == "__main__":
    train_dqn_rule()
