# Created by shaji at 12/02/2024
import time

import shutil
from tqdm import tqdm
from ocatari.core import OCAtari

import pi.utils.atari.game_patches
from pi.utils import game_utils, draw_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils.oc_utils import extract_logic_state_atari
from pi.utils.atari import game_patches
from src.utils_game import render_getout


def _render(args, agent, env_args, video_out):
    # render the game
    screen_text = (
        f"{agent.agent_type} ep: {env_args.game_i}, Rec: {env_args.best_score} act: {args.action_names[env_args.action]}")
    wr_plot = game_utils.plot_wr(env_args)
    mt_plot = game_utils.plot_mt_asterix(env_args, agent)
    video_out, _ = game_utils.plot_game_frame(env_args, video_out, env_args.obs, wr_plot, mt_plot, [], screen_text)


def _act(agent, env_args, env):
    # agent predict an action
    if agent.agent_type == "smp":
        env_args.action, env_args.explaining = agent.act(env.objects)
    elif agent.agent_type == "pretrained":
        env_args.action, _ = agent(env.dqn_obs.to(env_args.device))
    elif agent.agent_type == "random":
        env_args.action = agent.act(None)
    else:
        raise ValueError
    # env execute action
    env_args.last_obs, env_args.reward, env_args.terminated, env_args.truncated, info = env.step(env_args.action)
    ram = env._env.unwrapped.ale.getRAM()
    return info


def render_atari_game(agent, args, save_buffer):
    # args.m = args.m[0].upper() + args.m[1:]
    if args.device != "cpu":
        render_mode = None
    else:
        render_mode = 'rgb_array'
    env = OCAtari(args.m, mode="revised", hud=True, render_mode=render_mode)
    obs, info = env.reset()
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    agent.position_norm_factor = obs.shape[0]
    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)
    for game_i in tqdm(range(env_args.game_num), desc=f"Agent  {agent.agent_type}"):
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

            # agent predict an action
            env_args.logic_state, env_args.state_score = extract_logic_state_atari(env.objects, args.game_info,
                                                                                   obs.shape[0])
            env_args.obs = env_args.last_obs
            info = _act(agent, env_args, env)

            # if env_args.reward > 0:
            #     env_args.state_score += env_args.reward
            # if env_args.reward < 0:
            #     env_args.state_loss += env_args.reward
            # assign reward for lost one live
            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1

                env_args.update_lost_live(info["lives"])
                # revise the game rules
                if agent.agent_type == "smp" and len(env_args.logic_states) > 2 and game_i % args.reasoning_gap == 0:
                    agent.revise_loss(args, env_args)
                    if args.with_explain:
                        game_utils.revise_loss_log(env_args, agent, video_out)
                if args.save_frame:
                    # move dead frame to some folder
                    shutil.copy2(env_args.output_folder / "frames" / f"g_{env_args.game_i}_f_{env_args.frame_i}.png",
                                 env_args.output_folder / "lost_frames" / f"g_{env_args.game_i}_f_{env_args.frame_i}.png")
            else:
                # record game states
                env_args.buffer_frame()
                # render the game
                if args.with_explain:
                    _render(args, agent, env_args, video_out)

                game_utils.frame_log(agent, env_args)
            # update game args
            env_args.update_args()
        env_args.buffer_game(args.zero_reward, args.save_frame)

        env_args.reset_buffer_game()
        game_utils.game_over_log(args, agent, env_args)
        env_args.win_rate[game_i] = env_args.state_score  # update ep score
    env.close()
    game_utils.finish_one_run(env_args, args, agent)
    if save_buffer:
        game_utils.save_game_buffer(args, env_args)
    if args.with_explain:
        draw_utils.release_video(video_out)


def render_game(agent, args, save_buffer=False):
    if args.m == 'getout' or args.m == "getoutplus":
        render_getout(agent, args)
    elif args.m in ['Asterix', 'Boxing', 'Kangaroo']:
        render_atari_game(agent, args, save_buffer)
    else:
        raise ValueError("Game not exist.")
