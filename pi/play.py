# Created by jing at 01.12.23
import os.path
import time
import torch
from ocatari.core import OCAtari

from tqdm import tqdm
import os

from pi.utils.EnvArgs import EnvArgs
from pi.utils import args_utils
from src import config
from pi.utils.atari import game_patches
from pi.utils.oc_utils import extract_logic_state_atari

from pi.utils.game_utils import create_agent

from pi import train_utils
from pi.utils import game_utils, args_utils, reason_utils, draw_utils
from src import config

num_cores = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(1)


def _reason_action(args, agent, env, env_args, mlp_a, mlp_c, mlp_t):
    # determine relation related objects
    if args.m == "Asterix":
        obj_id = agent.select_action(env.dqn_obs.to(env_args.device))
        state_kinematic = reason_utils.extract_asterix_kinematics(args, env_args.past_states)
        collective_indices, collective_id_dqn = reason_utils.asterix_obj_to_collective(obj_id)
        with_target = True
    elif args.m == "Pong":
        with_target = False
        state_kinematic = reason_utils.extract_pong_kinematics(args, env_args.past_states)
        state_symbolic = reason_utils.extract_pong_symbolic(args, state_kinematic)
    elif args.m == "Kangaroo":
        with_target = True
        obj_id = agent.select_action(env.dqn_obs.to(env_args.device))
        state_kinematic = reason_utils.extract_kangaroo_kinematics(args, env_args.past_states)
        collective_indices, collective_id_dqn = reason_utils.kangaroo_obj_to_collective(obj_id + 1)
    else:
        raise ValueError

    # determin object types

    kinematic_series_data = train_utils.get_stack_buffer(state_kinematic, args.stack_num)
    input_c_tensor = kinematic_series_data[-1, 1:].reshape(1, -1)
    collective_id_mlp_conf = mlp_c(input_c_tensor)
    collective_id_mlp = collective_id_mlp_conf.argmax()
    # collective_id_mlp = 0
    # select mlp_a
    mlp_a_i = mlp_a[collective_id_mlp]
    collective_kinematic = kinematic_series_data[-1, collective_id_mlp + 1].unsqueeze(0)
    # determine action
    action = mlp_a_i(collective_kinematic.view(1, -1)).argmax()
    rule_data = reason_utils.get_rule_data(state_symbolic, collective_id_mlp + 1, action, args)
    return action, collective_id_mlp, rule_data


def main(render=True, m=None):
    # load arguments
    args = args_utils.load_args(config.path_exps, m)
    # learn behaviors from data
    student_agent = create_agent(args, agent_type='smp')
    # load MLP-A
    obj_type_num = len(args.game_info["obj_info"]) - 1
    mlp_a = train_utils.load_mlp_a(args, args.trained_model_folder, obj_type_num, args.m)
    # load MLP-C
    mlp_c = train_utils.load_mlp_c(args)
    # mlp_c = None
    mlp_t = train_utils.load_mlp_t(args)
    # Initialize environment
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    input_shape = env.observation_space.shape

    # Initialize agent
    env_args = EnvArgs(agent=student_agent, args=args, window_size=obs.shape[:2], fps=60)
    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)

    for game_i in tqdm(range(args.teacher_game_nums), desc=f"Describing agent  {student_agent.agent_type}"):
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
                obj_id = torch.tensor([[0]]).to(args.device)
                rule_data = None
            else:
                action, obj_id, rule_data = _reason_action(args, student_agent, env, env_args, mlp_a, mlp_c, mlp_t)

            state = env.dqn_obs.to(args.device)
            env_args.obs, env_args.reward, env_args.terminated, env_args.truncated, info = env.step(action)
            game_patches.atari_frame_patches(args, env_args, info)
            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, student_agent, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1
                env_args.update_lost_live(args.m, info["lives"])
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
                draw_utils.addCustomText(env_args.obs, f"Logic",
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
            if rule_data is not None:
                env_args.rule_data_buffer.append(rule_data)
            env_args.reward = torch.tensor(env_args.reward).reshape(1).to(args.device)

        # env_args.buffer_game(args.zero_reward, args.save_frame)
        reason_utils.reason_rules(env_args)
        env_args.win_rate[game_i] = sum(env_args.rewards[:-1])  # update ep score
        env_args.reset_buffer_game()
    env.close()
    game_utils.finish_one_run(env_args, args, student_agent)
    buffer_filename = args.game_buffer_path / f"learn_buffer_describing_{args.teacher_game_nums}.json"
    game_utils.save_game_buffer(args, env_args, buffer_filename)
    if args.with_explain:
        draw_utils.release_video(video_out)

    return student_agent


if __name__ == "__main__":
    main()
