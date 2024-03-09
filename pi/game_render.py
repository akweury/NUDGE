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


def _render(args, agent, env_args, video_out, agent_type):
    # render the game

    screen_text = (
        f"{agent.agent_type} ep: {env_args.game_i}, Rec: {env_args.best_score} \n "
        f"act: {args.action_names[env_args.action]} re: {env_args.reward}")

    if env_args.frame_i % 100 == 0:
        if agent.agent_type == "smp" and agent.model.pwt is not None:
            analysis_data = agent.model.pwt
        else:
            analysis_data = torch.zeros(10, 10)

        env_args.analysis_plot = draw_utils.plot_heat_map(analysis_data,
                                                          args.output_folder, "o2o_weights", figsize=(5, 5))

    # env_args.logic_state = agent.now_state
    video_out, _ = game_utils.plot_game_frame(agent_type, env_args, video_out, env_args.obs, env_args.analysis_plot,
                                              screen_text)


def _act(agent, env_args, env):
    # agent predict an action
    if agent.agent_type == "smp":

        if env_args.frame_i <= 20 or env_args.new_life:
            env_args.action = 1
            env_args.explaining = None
            env_args.new_life = False
        else:
            env_args.action, env_args.explaining = agent.act(env.objects)
            env_args.explain_text = env_args.explaining["text"]
    elif agent.agent_type == "pretrained":
        env_args.action, _ = agent(env.dqn_obs.to(env_args.device))
    elif agent.agent_type == "random":
        env_args.action = agent.act(None)
    else:
        raise ValueError
    # env execute action
    env_args.last_obs, env_args.reward, env_args.terminated, env_args.truncated, info = env.step(env_args.action)
    ram = env._env.unwrapped.ale.getRAM()
    if agent.agent_type == "smp":
        agent.model.last2nd_state = agent.model.last_state
        agent.model.last_state = torch.tensor(env_args.logic_state).unsqueeze(0)
    return info


def _update_weights(env_args, student_agent):
    if env_args.frame_i <= env_args.jump_frames:
        return
    student_agent.learn_from_dqn(env_args.action)


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
                    _render(args, agent, env_args, video_out, "")

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
    oc_name = game_utils.get_ocname(args.m)
    env = OCAtari(oc_name, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    agent.position_norm_factor = obs.shape[0]
    if agent.agent_type == "smp":
        agent.model.pwt = torch.load(args.o2o_weight_file)

    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)
    for game_i in tqdm(range(env_args.game_num), desc=f"Agent  {agent.agent_type}"):
        if agent.agent_type == "smp":
            agent.model.game_o2o_weights = []
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
            if env_args.frame_i == 250:
                print("")
            env_args.logic_state, env_args.state_score = extract_logic_state_atari(args, env.objects, args.game_info,
                                                                                   obs.shape[0])
            env_args.obs = env_args.last_obs

            info = _act(agent, env_args, env)
            game_patches.atari_frame_patches(args, env_args, info)
            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1
                env_args.update_lost_live(info["lives"])
                if args.save_frame:
                    # move dead frame to some folder
                    shutil.copy2(env_args.output_folder / "frames" / f"g_{env_args.game_i}_f_{env_args.frame_i}.png",
                                 env_args.output_folder / "lost_frames" / f"g_{env_args.game_i}_f_{env_args.frame_i}.png")
            else:
                # record game states
                env_args.next_state, env_args.state_score = extract_logic_state_atari(args, env.objects, args.game_info,
                                                                                      obs.shape[0])
                env_args.buffer_frame()
                # render the game
                if args.with_explain or args.save_frame:
                    _render(args, agent, env_args, video_out, agent.agent_type)

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


def train_atari_game(teacher_agent, student_agent, args, o2o_data):
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(agent=teacher_agent, args=args, window_size=obs.shape[:2], fps=60)
    teacher_agent.position_norm_factor = obs.shape[0]
    student_agent.position_norm_factor = obs.shape[0]
    video_out = game_utils.get_game_viewer(env_args)
    for game_i in tqdm(range(env_args.train_num), desc=f"Agent  {teacher_agent.agent_type}"):
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

            env_args.logic_state, _ = extract_logic_state_atari(env.objects, args.game_info, obs.shape[0])
            env_args.obs = env_args.last_obs
            student_agent.now_state = torch.tensor(env_args.logic_state).to(args.device)
            teacher_agent.now_state = torch.tensor(env_args.logic_state).to(args.device)
            info = _act(teacher_agent, env_args, env)

            next_state, _ = extract_logic_state_atari(env.objects, args.game_info, obs.shape[0])
            student_agent.next_state = torch.tensor(next_state).to(args.device)

            _update_weights(env_args, student_agent)
            # update the weight of student agent

            game_patches.atari_frame_patches(args, env_args, info)
            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1

                env_args.update_lost_live(info["lives"])
                # revise the game rules
                if teacher_agent.agent_type == "smp" and len(
                        env_args.logic_states) > 2 and game_i % args.reasoning_gap == 0 and args.revise:
                    teacher_agent.revise_loss(args, env_args)
                    if args.with_explain:
                        game_utils.revise_loss_log(env_args, teacher_agent, video_out)
                if args.save_frame:
                    # move dead frame to some folder
                    shutil.copy2(env_args.output_folder / "frames" / f"g_{env_args.game_i}_f_{env_args.frame_i}.png",
                                 env_args.output_folder / "lost_frames" / f"g_{env_args.game_i}_f_{env_args.frame_i}.png")
            else:
                # record game states
                env_args.buffer_frame()
                # render the game
                if args.with_explain or args.save_frame:
                    _render(args, student_agent, env_args, video_out, "dqn")

                    game_utils.frame_log(teacher_agent, env_args)
            # update game args
            # if env_args.explain_text != '':
            #     print("")
            env_args.update_args()
        env_args.buffer_game(args.zero_reward, args.save_frame)

        env_args.reset_buffer_game()
        game_utils.game_over_log(args, teacher_agent, env_args)
        env_args.win_rate[game_i] = env_args.state_score  # update ep score
    env.close()
    game_utils.finish_one_run(env_args, args, teacher_agent)

    torch.save(student_agent.model.pwt, args.o2o_weight_file)

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
    elif args.m in ["Assault", 'Asterix', 'Boxing', 'Kangaroo', "Breakout", "Freeway", "Pong", "Frostbite",
                    "montezuma_revenge", "fishing_derby"]:
        render_atari_game(agent, args, save_buffer)
    else:
        raise ValueError("Game not exist.")
