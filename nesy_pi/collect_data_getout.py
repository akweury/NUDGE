# Created by jing at 15.04.24
import os
import torch
import random
import numpy as np
from rtpt import RTPT
from tqdm import tqdm
from src import config
import time
from nesy_pi.aitk.utils.EnvArgs import EnvArgs

from nesy_pi.aitk.utils import args_utils, game_utils, draw_utils
from nesy_pi.aitk.utils.game_utils import RolloutBuffer
from src.environments.getout.getout.getout.getout import Getout
from src.environments.getout.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from src.agents.utils_getout import extract_logic_state_getout

args = args_utils.get_args()
rtpt = RTPT(name_initials='JS', experiment_name=f"{args.m}_{args.start_frame}_{args.end_frame}",
            max_iterations=args.end_frame - args.start_frame)
# Start the RTPT tracking
rtpt.start()
# learn behaviors from data
# collect game buffer from neural agent
buffer_filename = args.trained_model_folder / f"nesy_pi_{args.teacher_game_nums}.json"


def _render(args, agent, env_args, video_out, agent_type):
    # render the game

    screen_text = (
        f"ep: {env_args.game_i}\n "
        f"act: {args.action_names[env_args.action - 1]} re: {env_args.reward}")
    # env_args.logic_state = agent.now_state
    video_out, _ = game_utils.plot_game_frame(agent_type, env_args, video_out, env_args.obs,
                                              screen_text)


def create_getout_instance(args, seed=None):
    if args.m == 'getoutplus':
        enemies = True
    else:
        enemies = False
    # level_generator = DummyGenerator()
    getout = Getout()
    level_generator = ParameterizedLevelGenerator(enemies=enemies)
    level_generator.generate(getout, seed=seed)
    getout.render()

    return getout


def collect_getout_data(args, agent, buffer_filename, save_buffer):
    game_num = args.teacher_game_nums
    # play games using the random agent
    seed = random.seed()
    buffer = RolloutBuffer(buffer_filename)

    if os.path.exists(buffer.filename):
        return
    # collect data
    step = 0
    win_count = 0
    win_rates = []
    if args.m == 'getout':
        game_env = create_getout_instance(args, seed)
        env_args = EnvArgs(agent=agent, args=args, window_size=[game_env.camera.height,
                                                                game_env.camera.width], fps=60)

        if args.with_explain:
            video_out = game_utils.get_game_viewer(env_args)
        # frame rate limiting
        for i in tqdm(range(game_num), desc=f"win counter: {win_count}"):
            step += 1
            logic_states = []
            actions = []
            rewards = []
            frame_counter = 0

            # play a game
            while not (game_env.level.terminated):
                # limit frame rate
                if args.with_explain:
                    current_frame_time = time.time()
                    if env_args.last_frame_time + env_args.target_frame_duration > current_frame_time:
                        sl = (env_args.last_frame_time + env_args.target_frame_duration) - current_frame_time
                        time.sleep(sl)
                        continue
                    env_args.last_frame_time = current_frame_time  # save frame start time for next iteration

                env_args.obs = env_args.last_obs
                # random actions
                action = agent.reasoning_act(game_env)
                logic_state = extract_logic_state_getout(game_env, args).squeeze()
                env_args.logic_state = logic_state
                try:
                    reward = game_env.step(action)
                except KeyError:
                    game_env.level.terminated = True
                    game_env.level.lost = True
                    break
                if frame_counter == 0:
                    logic_states.append(logic_state.detach().tolist())
                    actions.append(action - 1)
                    rewards.append(reward)
                else:
                    logic_states.append(logic_state.detach().tolist())
                    actions.append(action - 1)
                    rewards.append(reward)
                env_args.last_obs = np.array(game_env.camera.screen.convert("RGB"))
                frame_counter += 1

                if args.with_explain:
                    env_args.frame_i = frame_counter
                    env_args.game_i = i
                    env_args.action = action
                    env_args.reward = reward
                    _render(args, agent, env_args, video_out, "")
                if frame_counter > 100:
                    game_env.level.terminated = True
                    game_env.level.lost = True
            # save game buffer
            if not game_env.level.lost and len(logic_states) < 100:
                buffer.logic_states.append(logic_states)
                buffer.actions.append(actions)
                buffer.rewards.append(rewards)
                win_count += 1

            else:
                buffer.lost_logic_states.append(logic_states)
                buffer.lost_actions.append(actions)
                buffer.lost_rewards.append(rewards)

            win_rates.append(win_count / (i + 1e-20))
            # start a new game
            game_env = create_getout_instance(args, seed)

        buffer.win_rates = win_rates
        buffer.save_data()
    if args.with_explain:
        draw_utils.release_video(video_out)
    return


if not os.path.exists(buffer_filename):
    agent = game_utils.create_agent(args, agent_type='ppo')
    collect_getout_data(args, agent, buffer_filename, save_buffer=True)

game_buffer = game_utils.load_buffer(args, buffer_filename)
data_file = args.trained_model_folder / f"nesy_data.pth"
if not os.path.exists(data_file):
    states = torch.cat(game_buffer.logic_states, dim=0)
    states[:, :, -2:] = states[:, :, -2:] / 50
    actions = torch.cat(game_buffer.actions, dim=0)
    args.num_actions = len(actions.unique())
    data = {}
    for a_i in range(args.num_actions):
        action_mask = actions == a_i
        pos_data = states[action_mask]
        neg_data = states[~action_mask]
        data[a_i] = {"pos_data": pos_data, "neg_data": neg_data}
    torch.save(data, data_file)
    print(f"Saved data to {data_file}.")
