# Created by shaji at 12/02/2024
import time

import torch
from PIL import ImageDraw, Image
from tqdm import tqdm
from ocatari.core import OCAtari

from pi.utils.game_utils import RolloutBuffer
from pi.utils import game_utils, draw_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils.oc_utils import extract_logic_state_atari
from src.utils_game import render_getout


def _render(agent, env_args, video_out):
    # render the game
    screen_text = f"{agent.agent_type} ep: {env_args.game_i}, Rec: {env_args.best_score}"
    wr_plot = game_utils.plot_wr(env_args)
    mt_plot = game_utils.plot_mt_asterix(env_args, agent)
    video_out, _ = game_utils.plot_game_frame(env_args, video_out, env_args.obs, wr_plot, mt_plot, [], screen_text)


def _act(agent, env_args, env):
    # agent predict an action
    if agent.agent_type == "smp":
        env_args.action, env_args.explaining = agent.act(env.objects)
    elif agent.agent_type == "pretrained":
        env_args.action, _ = agent(env.dqn_obs)
    else:
        raise ValueError
    # env execute action
    env_args.obs, env_args.reward, terminated, truncated, info = env.step(env_args.action)
    ram = env._env.unwrapped.ale.getRAM()
    return info


def render_asterix(agent, args, save_buffer):
    # args.m = args.m[0].upper() + args.m[1:]
    print(f"game name: {args.m}")
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(args=args, window_size=obs.shape[:2], fps=60)
    video_out = game_utils.get_game_viewer(env_args)
    for game_i in tqdm(range(env_args.game_num)):
        env_args.obs, info = env.reset()
        env_args.reset_args(game_i)
        env_args.reset_buffer_game()
        while env_args.current_lives > 0:
            # limit frame rate
            if args.with_explain:
                current_frame_time = time.time()
                if env_args.last_frame_time + env_args.target_frame_duration > current_frame_time:
                    sl = (env_args.last_frame_time + env_args.target_frame_duration) - current_frame_time
                    time.sleep(sl)
                    continue
                env_args.last_frame_time = current_frame_time  # save frame start time for next iteration
            # agent predict an action
            info = _act(agent, env_args, env)
            env_args.logic_state, env_args.state_score = extract_logic_state_atari(env.objects, args.game_info)
            # assign reward for lost one live
            if info["lives"] < env_args.current_lives:
                if args.m == "Asterix":
                    env_args.logic_states, env_args.actions, env_args.rewards = game_utils.atari_patches(args.game_info,
                                                                                                         env_args.logic_states,
                                                                                                         env_args.actions,
                                                                                                         env_args.rewards)

                env_args.update_lost_live(info["lives"])
                # revise the game rules
                if agent.agent_type == "smp" and len(env_args.logic_states) > 2:
                    agent.revise_loss(args, env_args)
                    game_utils.revise_loss_log(env_args, agent, video_out)
                env_args.buffer_game()
                env_args.reset_buffer_game()
            else:
                # record game states
                env_args.buffer_frame()
            # render the game
            if args.with_explain:
                _render(agent, env_args, video_out)
                game_utils.frame_log(agent, env_args)
            env_args.update_args(env_args)

        game_utils.game_over_log(agent, env_args)
        env_args.win_rate[game_i] = env_args.state_score  # update ep score
    env.close()
    draw_utils.release_video(video_out)
    game_utils.finish_one_run(env_args, args, agent)

    if save_buffer:
        game_utils.save_game_buffer(args, env_args)


def render_game(agent, args, save_buffer=False):
    if args.m == 'getout' or args.m == "getoutplus":
        render_getout(agent, args)
    elif args.m == 'Assault':
        render_assault(agent, args)
    elif args.m == 'asterix':
        render_asterix(agent, args, save_buffer)
    elif args.m == "Kangaroo":
        render_asterix(agent, args, save_buffer)
    else:
        raise ValueError("Game not exist.")


def render_assault(agent, args):
    env = OCAtari(args.m, mode="vision", hud=True, render_mode='rgb_array')
    game_num = 300
    game_i = 0
    win_counter = 0
    win_rate = torch.zeros(2, game_num)
    win_rate[1, :] = agent.buffer_win_rates[:game_num]
    observation, info = env.reset()
    zoom_in = args.zoom_in
    viewer = game_utils.setup_image_viewer(env.game_name, zoom_in * observation.shape[0],
                                           zoom_in * observation.shape[1])

    # frame rate limiting
    fps = 60
    target_frame_duration = 1 / fps
    last_frame_time = 0

    explaining = {}
    while game_i < game_num:
        frame_i = 0
        terminated = False
        truncated = False
        decision_history = []
        current_lives = args.max_lives

        while not terminated or truncated:
            current_frame_time = time.time()
            # limit frame rate
            if last_frame_time + target_frame_duration > current_frame_time:
                sl = (last_frame_time + target_frame_duration) - current_frame_time
                time.sleep(sl)
                continue
            last_frame_time = current_frame_time  # save frame start time for next iteration

            if args.agent_type == "smp":
                action, explaining = agent.act(env.objects)
            elif args.agent_type == "ppo":
                action, _ = agent(env.dqn_obs)
            else:
                raise ValueError

            obs, reward, terminated, truncated, info = env.step(action)
            if info['lives'] < current_lives:
                reward = args.reward_lost_one_live
                current_lives = info['lives']

            # logging
            # for beh_i in explaining['behavior_index']:
            #     print(f"f: {game_i}, rw: {reward}, act: {action}, behavior: {agent.behaviors[beh_i].clause}")

            # visualization
            if frame_i % 5 == 0:
                if args.render:
                    ImageDraw.Draw(Image.fromarray(obs)).text((40, 60), f"ep: {game_i}, win: {win_counter}",
                                                              (120, 20, 20))
                    zoom_obs = game_utils.zoom_image(obs, zoom_in * observation.shape[0],
                                                     zoom_in * observation.shape[1])
                    viewer.show(zoom_obs[:, :, :3])

            if reward < 0:
                game_i += 1
                frame_i = 0
                win_rate[0, game_i] = win_counter / (game_i + 1e-20)

            elif reward > 0:
                win_counter += 1
                game_i += 1
                frame_i = 0
                win_rate[0, game_i] = win_counter / (game_i + 1e-20)

            ram = env._env.unwrapped.ale.getRAM()
            if explaining is not None:
                explaining["reward"].append(reward)
                decision_history.append(explaining)

            # finish one game
            if terminated or truncated:
                observation, info = env.reset()

            # the game total frame number has to greater than 2
            if len(decision_history) > 2 and reward < 0:
                lost_game_data = agent.revise_loss(decision_history)
                agent.update_lost_buffer(lost_game_data)
                def_behaviors = agent.reasoning_def_behaviors(use_ckp=False)
                agent.update_behaviors(None, def_behaviors, None, args)
                decision_history = []
                print("- revise loss finished.")
            frame_i += 1
            draw_utils.plot_line_chart(win_rate[:, :game_i], args.output_folder, ['smp', 'ppo'], title='win_rate',
                                       cla_leg=True)
        # modify and display render
    env.close()
