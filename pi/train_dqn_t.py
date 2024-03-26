# Created by shaji at 26/03/2024
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
from pi import train_utils


def train_dqn_t():
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

    if args.m == "Pong":
        # pos_data, actions = student_agent.pong_reasoner()
        num_obj_types = 2
    elif args.m == "Asterix":
        # pos_data, actions = student_agent.asterix_reasoner()
        num_obj_types = 2
    elif args.m == "Kangaroo":
        # pos_data, actions = student_agent.kangaroo_reasoner()
        num_obj_types = 8
    else:
        raise ValueError

    # load MLP-A
    mlp_a = []
    for obj_i in range(num_obj_types):
        mlp_a_i_file = args.trained_model_folder / f"{args.m}_{obj_i}.pth.tar"
        mlp_a_i = torch.load(mlp_a_i_file)["model"]
        mlp_a.append(mlp_a_i)

    # Initialize environment
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    num_actions = env.action_space.n
    input_shape = env.observation_space.shape

    # Initialize agent
    agent = train_utils.DQNAgent(args, input_shape, num_obj_types)
    agent.agent_type = "DQN-T"
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    env_args.learn_performance = []
    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)
    for game_i in tqdm(range(args.episode_num), desc=f"Agent  {agent.agent_type}"):
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

            obj_id = agent.select_action(env.dqn_obs.to(env_args.device))
            logic_state, _ = extract_logic_state_atari(args, env.objects, args.game_info, obs.shape[0])
            env_args.past_states.append(logic_state)
            if args.m == "Asterix":
                action = reason_utils.pred_asterix_action(args, env_args, env_args.past_states, obj_id + 1,
                                                          mlp_a[obj_id]).to(torch.int64).reshape(1)
            elif args.m == "Pong":
                action = reason_utils.pred_pong_action(args, env_args, env_args.past_states, obj_id + 1,
                                                       mlp_a[obj_id]).to(torch.int64).reshape(1)
            elif args.m == "Kangaroo":
                action = reason_utils.pred_kangaroo_action(args, env_args, env_args.past_states, obj_id + 1,
                                                           mlp_a[obj_id]).to(torch.int64).reshape(1)
            else:
                raise ValueError

            state = env.dqn_obs.to(args.device)
            env_args.obs, env_args.reward, env_args.terminated, env_args.truncated, info = env.step(action)
            game_patches.atari_frame_patches(args, env_args, info)
            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, agent, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1
                env_args.update_lost_live(args.m, info["lives"])
                if sum(env_args.rewards) > 0:
                    print('')
            if args.with_explain:
                screen_text = (
                    f"dqn_obj ep: {env_args.game_i}, Rec: {env_args.best_score} \n "
                    f"obj: {args.row_names[obj_id + 1]}, act: {args.action_names[action]} re: {env_args.reward}")
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
            # update game args
            env_args.update_args()
            env_args.rewards.append(env_args.reward)
            env_args.reward = torch.tensor(env_args.reward).reshape(1).to(args.device)
            next_state = env.dqn_obs.to(args.device) if not env_args.terminated else None
            # Store the transition in memory
            agent.memory.push(state, obj_id, next_state, env_args.reward, env_args.terminated)
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
            env_args.learn_performance.append(sum(env_args.win_rate[game_i - args.print_freq:game_i]))

            line_chart_data = torch.tensor(env_args.learn_performance)
            draw_utils.plot_line_chart(line_chart_data.unsqueeze(0), path=args.trained_model_folder,
                                       labels=[f"total_score_every_{args.print_freq}"],
                                       title=f"{args.m}_sum_past_{args.print_freq}",
                                       figure_size=(30, 5))
            # save model
            last_epoch_save_path = args.trained_model_folder / f'dqn_t_{game_i + 1 - args.print_freq}.pth'
            save_path = args.trained_model_folder / f'dqn_t_{game_i + 1}.pth'
            if os.path.exists(last_epoch_save_path):
                os.remove(last_epoch_save_path)
            from pi.utils import file_utils

            file_utils.save_agent(save_path, agent)

    env.close()
    game_utils.finish_one_run(env_args, args, agent)
    game_utils.save_game_buffer(args, env_args)
    if args.with_explain:
        draw_utils.release_video(video_out)


if __name__ == "__main__":
    train_dqn_t()
