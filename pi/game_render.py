# Created by shaji at 12/02/2024

import time
import numpy as np
import shutil
import torch
from tqdm import tqdm
from ocatari.core import OCAtari
from src.environments.getout.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from src.environments.getout.getout.getout.getout import Getout
from src.agents.utils_getout import extract_logic_state_getout

from pi.utils import game_utils, file_utils, draw_utils, math_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils.oc_utils import extract_logic_state_atari
from pi.utils.atari import game_patches
from pi.utils import reason_utils


def _render(args, agent, env_args, video_out):
    # render the game

    screen_text = (
        f"{agent.agent_type} ep: {env_args.game_i}, Rec: {env_args.best_score} \n "
        f"act: {args.action_names[env_args.action]} re: {env_args.reward}")
    wr_plot = game_utils.plot_wr(env_args)
    mt_plot = game_utils.plot_mt_asterix(env_args, agent)
    video_out, _ = game_utils.plot_game_frame(env_args, video_out, env_args.obs, wr_plot, mt_plot, [],
                                              screen_text)


def _act(agent, env_args, env):
    # agent predict an action
    if agent.agent_type == "smp":
        env_args.action, env_args.explaining = agent.act(env.objects)
        if env_args.frame_i == 0 or env_args.new_life:
            env_args.action = 1
            env_args.new_life = False
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


def render_getout(agent, args, save_buffer):
    def create_getout_instance(args, seed=None):
        if args.hardness == 1:
            enemies = True
        else:
            enemies = False
        # level_generator = DummyGenerator()
        getout = Getout()
        level_generator = ParameterizedLevelGenerator(enemies=enemies)
        level_generator.generate(getout, seed=seed)
        getout.render()

        return getout

    env = create_getout_instance(args)
    env_args = EnvArgs(agent=agent, args=args, window_size=[env.camera.height, env.camera.width], fps=60)
    agent.position_norm_factor = env.camera.height
    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)
    for game_i in tqdm(range(env_args.game_num), desc=f"Agent  {agent.agent_type}"):
        env = create_getout_instance(args)
        env_args.reset_args(game_i)
        env_args.reset_buffer_game()
        while not env_args.game_over:
            if env_args.frame_i > 300:
                break
            # limit frame rate
            if args.with_explain:
                current_frame_time = time.time()
                if env_args.last_frame_time + env_args.target_frame_duration > current_frame_time:
                    sl = (env_args.last_frame_time + env_args.target_frame_duration) - current_frame_time
                    time.sleep(sl)
                    continue
                env_args.last_frame_time = current_frame_time  # save frame start time for next iteration

            # agent predict an action
            env_args.logic_state = extract_logic_state_getout(env, args).squeeze().tolist()

            env_args.obs = env_args.last_obs
            env_args.action = agent.reasoning_act(env)
            try:
                env_args.reward = env.step(env_args.action)
            except KeyError:
                env.level.terminated = True
                env.level.lost = True
                env_args.terminated = True
                env_args.truncated = True
                break
            env_args.action = env_args.action - 1
            env_args.last_obs = np.array(env.camera.screen.convert("RGB"))
            if env.level.terminated:
                env_args.frame_i = len(env_args.logic_states) - 1

                # revise the game rules
                if agent.agent_type == "smp" and len(
                        env_args.logic_states) > 2 and game_i % args.reasoning_gap == 0 and args.revise:
                    agent.revise_loss(args, env_args)
                    if args.with_explain:
                        game_utils.revise_loss_log(env_args, agent, video_out)
                if args.save_frame:
                    # move dead frame to some folder
                    shutil.copy2(env_args.output_folder / "frames" / f"g_{env_args.game_i}_f_{env_args.frame_i}.png",
                                 env_args.output_folder / "lost_frames" / f"g_{env_args.game_i}_f_{env_args.frame_i}.png")
                env_args.game_over = True

            else:
                # record game states
                env_args.buffer_frame()
                # render the game
                if args.with_explain:
                    _render(args, agent, env_args, video_out)

                    game_utils.frame_log(agent, env_args)
            # update game args
            env_args.update_args()
        if not env.level.lost:
            env_args.win_rate[game_i] = env_args.win_rate[game_i - 1] + 1
        else:
            env_args.win_rate[game_i] = env_args.win_rate[game_i - 1]
        env_args.state_score = env_args.win_rate[game_i - 1]
        env_args.buffer_game(args.zero_reward, args.save_frame)
        env_args.reset_buffer_game()
        game_utils.game_over_log(args, agent, env_args)

    game_utils.finish_one_run(env_args, args, agent)
    if save_buffer:
        game_utils.save_game_buffer(args, env_args)
    if args.with_explain:
        draw_utils.release_video(video_out)


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
            game_patches.atari_frame_patches(args, env_args, info)
            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1

                env_args.update_lost_live(info["lives"])
                # revise the game rules
                if agent.agent_type == "smp" and len(
                        env_args.logic_states) > 2 and game_i % args.reasoning_gap == 0 and args.revise:
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
                if args.with_explain or args.save_frame:
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


def replay_atari_game(agent, args, o2o_data):
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    agent.position_norm_factor = obs.shape[0]
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

            if env_args.last2nd_state is not None:
                env_args.explain_text = reason_utils.game_explain(env_args.logic_state,
                                                                  env_args.last_state,
                                                                  env_args.last2nd_state,
                                                                  o2o_data)

            env_args.obs = env_args.last_obs
            env_args.last2nd_state = env_args.last_state
            env_args.last_state = env_args.logic_state
            info = _act(agent, env_args, env)
            game_patches.atari_frame_patches(args, env_args, info)
            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1

                env_args.update_lost_live(info["lives"])
                # revise the game rules
                if agent.agent_type == "smp" and len(
                        env_args.logic_states) > 2 and game_i % args.reasoning_gap == 0 and args.revise:
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
                if args.with_explain or args.save_frame:
                    _render(args, agent, env_args, video_out)

                    game_utils.frame_log(agent, env_args)
            # update game args
            # if env_args.explain_text != '':
            #     print("")
            env_args.update_args()
        env_args.buffer_game(args.zero_reward, args.save_frame)

        env_args.reset_buffer_game()
        game_utils.game_over_log(args, agent, env_args)
        env_args.win_rate[game_i] = env_args.state_score  # update ep score
    env.close()
    game_utils.finish_one_run(env_args, args, agent)
    game_utils.save_game_buffer(args, env_args)
    draw_utils.release_video(video_out)


def show_analysis_frame(frame_i, env_args, video_out, frame, pos, acc):
    if acc is not None:
        for obj_i in range(len(acc)):
            acc_text = [f"{n:.2f}" for n in acc[obj_i]]
            draw_utils.addCustomText(frame, str(acc_text), pos[obj_i], font_size=0.3, shift=[0, 20])
            frame = draw_utils.draw_arrow(frame, pos[obj_i], acc[obj_i], scale=5, shift=[30, 0])

    draw_utils.save_np_as_img(frame,
                              env_args.output_folder / "acc_frames" / f"acc_f_{frame_i}.png")
    out = draw_utils.write_video_frame(video_out, frame)


def render_game(agent, args, save_buffer=False):
    if args.m == 'getout' or args.m == "getoutplus":
        render_getout(agent, args, save_buffer)
    elif args.m in ["Assault", 'Asterix', 'Boxing', 'Kangaroo', "Breakout", "Freeway", "Pong"]:

        render_atari_game(agent, args, save_buffer)
    else:
        raise ValueError("Game not exist.")
