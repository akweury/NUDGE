# Created by jing at 15.04.24
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
# learn behaviors from data
# collect game buffer from neural agent
buffer_filename = args.trained_model_folder / f"nesy_pi_{args.teacher_game_nums}.json"
game_buffer = game_utils.load_buffer(args, buffer_filename)
data_file = args.trained_model_folder / f"nesy_data.pth"
if not os.path.exists(data_file):
    states = torch.cat(game_buffer.logic_states, dim=0)
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
