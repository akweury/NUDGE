# Created by jing at 15.04.24
import os
import torch
import random
from rtpt import RTPT
from tqdm import tqdm
from src import config

from nesy_pi.aitk.utils import args_utils, game_utils
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
    seed = random.seed() if args.seed is None else int(args.seed)
    args.filename = args.m + '_' + '_episode_' + str(game_num) + '.json'

    buffer = RolloutBuffer(args.filename)

    if os.path.exists(buffer.filename):
        return
    # collect data
    step = 0
    win_count = 0
    win_rates = []
    if args.m == 'getout':
        game_env = create_getout_instance(args)

        # frame rate limiting
        for i in tqdm(range(game_num), desc=f"win counter: {win_count}"):
            step += 1
            logic_states = []
            actions = []
            rewards = []
            frame_counter = 0

            # play a game
            while not (game_env.level.terminated):
                # random actions
                action = agent.reasoning_act(game_env)
                logic_state = extract_logic_state_getout(game_env, args).squeeze()
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
                elif action - 1 != actions[-1] or frame_counter % 5 == 0:
                    logic_states.append(logic_state.detach().tolist())
                    actions.append(action - 1)
                    rewards.append(reward)

                frame_counter += 1

            # save game buffer
            if not game_env.level.lost:
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
            game_env = create_getout_instance(args)

        buffer.win_rates = win_rates
        buffer.save_data()

    return


if not os.path.exists(buffer_filename):
    agent = game_utils.create_agent(args, agent_type='ppo')
    collect_getout_data(args, agent, buffer_filename, save_buffer=True)

game_buffer = game_utils.load_buffer(args, buffer_filename)
data_file = args.trained_model_folder / f"nesy_data.pth"
if not os.path.exists(data_file):
    states = torch.cat(game_buffer.logic_states, dim=0)
    actions = torch.cat(game_buffer.actions, dim=0)
    args.num_actions = len(actions.unique())
    data = {}
    for a_i in range(args.num_actions):
        action_mask = actions == a_i
        states[:, :, -2:] = states[:, :, -2:] / 50
        pos_data = states[action_mask]
        neg_data = states[~action_mask]
        data[a_i] = {"pos_data": pos_data, "neg_data": neg_data}
    torch.save(data, data_file)
    print(f"Saved data to {data_file}.")
