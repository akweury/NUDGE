# Created by shaji at 13/04/2024

import os
import torch
import time
from rtpt import RTPT
import copy
from ocatari.core import OCAtari
from nesy_pi.aitk.utils import log_utils
from tqdm import tqdm
from src import config

from nesy_pi.aitk.utils import args_utils, file_utils, game_utils

args = args_utils.get_args()
rtpt = RTPT(name_initials='JS', experiment_name=f"{args.m}_{args.start_frame}_{args.end_frame}",
            max_iterations=args.end_frame - args.start_frame)
# Start the RTPT tracking
rtpt.start()
# Initialize environment
env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
obs, info = env.reset()
dqn_a_input_shape = env.observation_space.shape
args.num_actions = env.action_space.n
agent = game_utils.create_agent(args, agent_type=args.teacher_agent)
args.log_file = log_utils.create_log_file(args.trained_model_folder, "pi")
# learn behaviors from data
# collect game buffer from neural agent
buffer_filename = args.trained_model_folder / f"nesy_pi_{args.teacher_game_nums}.json"
if not os.path.exists(buffer_filename):
    game_utils.collect_data_dqn_a(agent, args, buffer_filename, save_buffer=True)
game_buffer = game_utils.load_atari_buffer(args, buffer_filename)

data_file = args.trained_model_folder / f"nesy_data.pth"
if not os.path.exists(data_file):
    states = torch.cat([state for state in game_buffer["states"]], dim=0)
    actions = torch.cat([action for action in game_buffer["actions"]], dim=0)
    data = {}
    for a_i in range(args.num_actions):
        action_mask = actions == a_i
        pos_data = states[action_mask][:,[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        neg_data = states[~action_mask][:,[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        data[a_i] = {"pos_data": pos_data, "neg_data": neg_data}
    torch.save(data, data_file)
    print(f"Saved data to {data_file}.")
