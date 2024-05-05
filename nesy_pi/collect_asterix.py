# Created by shaji at 02/05/2024


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
    state_len = torch.tensor([len(state) for state in game_buffer["states"]])
    idx = state_len.sort()[1][-20:]
    states = torch.cat([game_buffer["states"][i][:-10] for i in idx], dim=0)
    actions = torch.cat([game_buffer["actions"][i][:-10] for i in idx], dim=0)

    random_indices = torch.randperm(states.shape[0])
    states = states[random_indices]
    actions = actions[random_indices]
    stored_data = {}
    for a_i in range(args.num_actions):
        action_mask = actions == a_i
        pos_data = states[action_mask]

        # positive data
        player_pos_data = pos_data[:, 0:1]
        other_objs = pos_data[:, 1:]
        # same with player
        mask_same_y = (pos_data[:, 1:, -1] - pos_data[:, 0:1, -1]).abs() < 0.05
        pos_same_y = []
        for s_i in range(len(other_objs)):
            _, above_indices = (player_pos_data[s_i, 0, -1] - other_objs[s_i, mask_same_y[s_i], -1]).abs().sort()
            data = other_objs[s_i][mask_same_y[s_i]][above_indices][:1]
            if data.shape[0] < 1:
                data = torch.cat([torch.zeros(1 - data.shape[0], 9).to(args.device), data], dim=0)
            pos_same_y.append(data.unsqueeze(0))
        pos_same_y = torch.cat(pos_same_y, dim=0)
        # above of player
        mask_above = (pos_data[:, 1:, -1] - pos_data[:, 0:1, -1]) < -0.05
        pos_above_data = []
        for s_i in range(len(other_objs)):
            _, above_indices = (player_pos_data[s_i, 0, -1] - other_objs[s_i, mask_above[s_i], -1]).abs().sort()
            data = other_objs[s_i][mask_above[s_i]][above_indices][:1]
            if data.shape[0] < 1:
                data = torch.cat([torch.zeros(1 - data.shape[0], 9).to(args.device), data], dim=0)
            pos_above_data.append(data.unsqueeze(0))
        pos_above_data = torch.cat(pos_above_data, dim=0)

        mask_below = (pos_data[:, 1:, -1] - pos_data[:, 0:1, -1]) > 0.05
        pos_below_data = []
        for s_i in range(len(other_objs)):
            _, below_indices = (-player_pos_data[s_i, 0, -1] + other_objs[s_i, mask_below[s_i], -1]).abs().sort()
            data = other_objs[s_i][mask_below[s_i]][below_indices][:1]
            if data.shape[0] < 1:
                data = torch.cat([torch.zeros(1 - data.shape[0], 9).to(args.device), data], dim=0)
            pos_below_data.append(data.unsqueeze(0))
        pos_below_data = torch.cat(pos_below_data, dim=0)

        neg_data = states[~action_mask]
        player_neg_data = neg_data[:, 0:1, ]
        other_objs = neg_data[:, 1:]
        mask_same_y_neg = (neg_data[:, 1:, -1] - neg_data[:, 0:1, -1]).abs() < 0.05
        neg_same_y = []
        for s_i in range(len(other_objs)):
            _, above_indices = (player_neg_data[s_i, 0, -1] - other_objs[s_i, mask_same_y_neg[s_i], -1]).abs().sort()
            data = other_objs[s_i][mask_same_y_neg[s_i]][above_indices][:1]
            if data.shape[0] < 1:
                data = torch.cat([torch.zeros(1 - data.shape[0], 9).to(args.device), data], dim=0)
            neg_same_y.append(data.unsqueeze(0))
        neg_same_y = torch.cat(neg_same_y, dim=0)

        neg_above_data = []
        mask_above = (neg_data[:, 1:, -1] - neg_data[:, 0:1, -1]) < -0.05
        for s_i in range(len(other_objs)):
            _, above_indices = (player_neg_data[s_i, 0, -1] - other_objs[s_i, mask_above[s_i], -1]).abs().sort()
            data = other_objs[s_i][mask_above[s_i]][above_indices][:1]
            if data.shape[0] < 1:
                data = torch.cat([torch.zeros(1 - data.shape[0], 9).to(args.device), data], dim=0)
            neg_above_data.append(data.unsqueeze(0))
        neg_above_data = torch.cat(neg_above_data, dim=0)

        mask_below = (neg_data[:, 1:, -1] - neg_data[:, 0:1, -1]) > 0.05
        neg_below_data = []
        for s_i in range(len(other_objs)):
            _, below_indices = (-player_neg_data[s_i, 0, -1] + other_objs[s_i, mask_below[s_i], -1]).abs().sort()
            data = other_objs[s_i][mask_below[s_i]][below_indices][:1]
            if data.shape[0] < 1:
                data = torch.cat([torch.zeros(1 - data.shape[0], 9).to(args.device), data], dim=0)
            neg_below_data.append(data.unsqueeze(0))
        neg_below_data = torch.cat(neg_below_data, dim=0)

        pos_ab_data = torch.cat((player_pos_data, pos_same_y, pos_above_data, pos_below_data), dim=1)
        neg_ab_data = torch.cat((player_neg_data, neg_same_y, neg_above_data, neg_below_data), dim=1)
        # existence of above 1 and below 1
        # existence of above 2 and below 1

        stored_data[a_i] = {}
        stored_data[a_i] = {"pos_data": pos_ab_data, "neg_data": neg_ab_data}


    torch.save(stored_data, data_file)
    print(f"Saved data to {data_file}.")
