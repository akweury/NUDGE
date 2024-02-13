# Created by shaji at 12/02/2024
import time

import torch
from PIL import ImageDraw, Image

from ocatari import OCAtari

from pi.utils import game_utils, draw_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils.oc_utils import extract_logic_state_asterix, extract_logic_state_atari
from src.utils_game import render_getout


def render_asterix(agent, args):
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(args=args, game_num=300, window_size=obs.shape[:2], fps=60)
    video_out = game_utils.get_game_viewer(env_args)
    explaining = None
    db_plots = []
    dead_counter = 0

    while env_args.game_i < env_args.game_num:
        obs, info = env.reset()

        frame_i = 0
        logic_states = []
        actions = []
        rewards = []
        consume_counter = 0
        env_args.current_lives = args.max_lives
        while env_args.current_lives > 0:
            current_frame_time = time.time()
            # limit frame rate
            if env_args.last_frame_time + env_args.target_frame_duration > current_frame_time:
                sl = (env_args.last_frame_time + env_args.target_frame_duration) - current_frame_time
                time.sleep(sl)
                continue
            env_args.last_frame_time = current_frame_time  # save frame start time for next iteration
            if agent.agent_type == "smp":
                action, explaining = agent.act(env.objects)
            elif agent.agent_type == "pretrained":
                action, _ = agent(env.dqn_obs)
            else:
                raise ValueError
            obs, reward, terminated, truncated, info = env.step(action)
            ram = env._env.unwrapped.ale.getRAM()
            logic_state, state_score = extract_logic_state_atari(env.objects, args.game_info)
            if state_score > env_args.best_score:
                env_args.best_score = state_score
            # assign reward for lost one live
            if info["lives"] < env_args.current_lives:
                reward += env_args.reward_lost_one_live
                env_args.current_lives = info["lives"]
                env_args.score_update = True
                env_args.win_rate[0, env_args.game_i] = env_args.best_score
                rewards[-1] += env_args.reward_lost_one_live

                # revise the game rules
                if len(logic_states) > 2:
                    dead_counter += 1
                    logic_states, actions, rewards = game_utils.asterix_patches(logic_states, actions, rewards)
                    agent.update_lost_buffer(logic_states, actions, rewards)
                    def_behaviors, db_plots = agent.reasoning_def_behaviors(use_ckp=False)
                    agent.update_behaviors(None, def_behaviors, None, args)
                    screen_text = f"ep: {env_args.game_i}, Rec: {env_args.best_score}"
                    game_utils.screen_shot(env_args, video_out, obs, wr_plot, mt_plot, db_plots, dead_counter,
                                           screen_text)
                logic_states = []
                actions = []
                rewards = []
                consume_counter = 0
                env_args.game_i += 1
            else:
                # record game states
                logic_states.append(logic_state)
                actions.append(action)
                rewards.append(reward)
                if reward > 0:
                    consume_counter += 1

            # render the game
            screen_text = f"ep: {env_args.game_i}, Rec: {env_args.best_score}"
            wr_plot = game_utils.plot_wr(env_args)
            mt_plot = game_utils.plot_mt_asterix(env_args, agent)
            video_out, _ = game_utils.plot_game_frame(env_args, video_out, obs, wr_plot, mt_plot, db_plots, screen_text)

            # game log
            try:
                for beh_i in explaining['behavior_index']:
                    print(
                        f"f: {frame_i}, rw: {reward}, act: {action - 1}, behavior: {agent.behaviors[beh_i].clause}")
            except IndexError:
                print("")
            # print(f"g: {env_args.game_i}, f: {frame_i}, rw: {reward}, act: {action}, lives:{info['lives']}, "
            #       f"agent: {int(torch.tensor(logic_state)[:, 0].sum())}, "
            #       f"enemy: {int(torch.tensor(logic_state)[:, 1].sum())}, "
            #       f"cauldron: {int(torch.tensor(logic_state)[:, 2].sum())}")

            frame_i = game_utils.update_game_args(frame_i, env_args, reward)

    env.close()
    draw_utils.release_video(video_out)


def render_kangaroo(agent, args):
    env = OCAtari(args.m, mode="vision", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(args=args, game_num=300, window_size=obs.shape[:2], fps=60)
    video_out = game_utils.get_game_viewer(env_args)
    explaining = None
    db_plots = []
    while env_args.game_i < env_args.game_num:
        frame_i = 0
        terminated = False
        truncated = False
        decision_history = []
        reward = 0
        current_lives = args.max_lives
        while not terminated or truncated:
            current_frame_time = time.time()
            # limit frame rate
            if env_args.last_frame_time + env_args.target_frame_duration > current_frame_time:
                sl = (env_args.last_frame_time + env_args.target_frame_duration) - current_frame_time
                time.sleep(sl)
                continue
            env_args.last_frame_time = current_frame_time  # save frame start time for next iteration
            if agent.agent_type == "smp":
                action, explaining = agent.act(env.objects)
            elif agent.agent_type == "pretrained":
                action, _ = agent(env.dqn_obs)
            else:
                raise ValueError
            obs, reward, terminated, truncated, info = env.step(action)

            ram = env._env.unwrapped.ale.getRAM()
            reward = game_utils.kangaroo_patches(env_args, reward, info["lives"])
            logic_state, state_score = extract_logic_state_atari(env.objects, args.game_info)
            if explaining is not None:
                explaining["reward"].append(reward)
                decision_history.append(explaining)
            # render the game
            wr_plot = game_utils.plot_wr(env_args)
            mt_plot = game_utils.plot_mt_asterix(env_args, agent)
            video_out, _ = game_utils.plot_game_frame(env_args, video_out, obs, wr_plot, mt_plot, db_plots)

            frame_i = game_utils.update_game_args(frame_i, env_args, reward)

        # finish one game
        if terminated or truncated:
            env.reset()

        # revise the game rules
        if len(decision_history) > 2 and reward < 0:
            lost_game_data = agent.revise_loss(decision_history)
            agent.update_lost_buffer(lost_game_data)
            def_behaviors, db_plots = agent.reasoning_def_behaviors(use_ckp=False)
            agent.update_behaviors(None, def_behaviors, None, args)
            print("- revise loss finished.")
    env.close()
    draw_utils.release_video(video_out)


def render_game(agent, args):
    if args.m == 'getout' or args.m == "getoutplus":
        render_getout(agent, args)
    elif args.m == 'Assault':
        render_assault(agent, args)
    elif args.m == 'Asterix':
        render_asterix(agent, args)
    elif args.m == "Kangaroo":
        render_kangaroo(agent, args)
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
