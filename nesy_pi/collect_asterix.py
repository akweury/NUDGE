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
    states = torch.cat([state for state in game_buffer["states"]], dim=0)
    actions = torch.cat([action for action in game_buffer["actions"]], dim=0)

    random_indices = torch.randperm(states.shape[0])
    states = states[random_indices]
    actions = actions[random_indices]
    stored_data = {}
    for a_i in range(args.num_actions):
        action_mask = actions == a_i
        pos_data = states[action_mask][:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

        # positive data
        player_pos_data = pos_data[:, 0:1]
        cars_data = pos_data[:, 1:]
        # above of player
        mask_above = (pos_data[:, 1:, -1] < pos_data[:, 0:1, -1])
        pos_above_data = []
        for s_i in range(len(cars_data)):
            _, above_indices = (player_pos_data[s_i, 0, -1] - cars_data[s_i, mask_above[s_i], -1]).sort()
            data = cars_data[s_i][mask_above[s_i]][above_indices][:3]
            if data.shape[0] < 3:
                data = torch.cat([torch.zeros(3 - data.shape[0], 8), data], dim=0)
            pos_above_data.append(data.unsqueeze(0))
        pos_above_data = torch.cat(pos_above_data, dim=0)

        mask_below = pos_data[:, 1:, -1] > pos_data[:, 0:1, -1]
        pos_below_data = []
        for s_i in range(len(cars_data)):
            _, below_indices = (-player_pos_data[s_i, 0, -1] + cars_data[s_i, mask_below[s_i], -1]).sort()
            data = cars_data[s_i][mask_below[s_i]][below_indices][:1]
            if data.shape[0] < 1:
                data = torch.cat([torch.zeros(1 - data.shape[0], 8), data], dim=0)
            pos_below_data.append(data.unsqueeze(0))
        pos_below_data = torch.cat(pos_below_data, dim=0)

        neg_data = states[~action_mask][:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        neg_above_data = []
        player_neg_data = neg_data[:, 0:1, ]
        cars_data = neg_data[:, 1:]
        mask_above = neg_data[:, 1:, -1] < neg_data[:, 0:1, -1]
        for s_i in range(len(cars_data)):
            _, above_indices = (player_neg_data[s_i, 0, -1] - cars_data[s_i, mask_above[s_i], -1]).sort()
            data = cars_data[s_i][mask_above[s_i]][above_indices][:3]
            if data.shape[0] < 3:
                data = torch.cat([torch.zeros(3 - data.shape[0], 8), data], dim=0)
            neg_above_data.append(data.unsqueeze(0))
        neg_above_data = torch.cat(neg_above_data, dim=0)

        mask_below = neg_data[:, 1:, -1] > neg_data[:, 0:1, -1]
        neg_below_data = []
        for s_i in range(len(cars_data)):
            _, below_indices = (-player_neg_data[s_i, 0, -1] + cars_data[s_i, mask_below[s_i], -1]).sort()
            data = cars_data[s_i][mask_below[s_i]][below_indices][:1]
            if data.shape[0] < 1:
                data = torch.cat([torch.zeros(1 - data.shape[0], 8), data], dim=0)
            neg_below_data.append(data.unsqueeze(0))
        neg_below_data = torch.cat(neg_below_data, dim=0)

        pos_ab_data = torch.cat((player_pos_data, pos_above_data, pos_below_data), dim=1)
        neg_ab_data = torch.cat((player_neg_data, neg_above_data, neg_below_data), dim=1)
        # existence of above 1 and below 1
        # existence of above 2 and below 1

        stored_data[a_i] = {}
        # existence of above 3 and below 1
        # a3b1_pos_mask = pos_ab_data[:, :, :2].sum(dim=-1).sum(dim=-1) == 5
        # a3b1_neg_mask = neg_ab_data[:, :, :2].sum(dim=-1).sum(dim=-1) == 5
        # pos_a3b1 = pos_ab_data[a3b1_pos_mask]
        # neg_a3b1 = neg_ab_data[a3b1_neg_mask]
        # # above 3 all greater than 0.5 (left to right direction)
        # a3b1_pos_mask_lr = (pos_a3b1[:, :, -1] > 0.5).sum(dim=-1) == 5
        # a3b1_neg_mask_lr = (neg_a3b1[:, :, -1] > 0.5).sum(dim=-1) == 5
        # pos_a3b1_lr = pos_a3b1[a3b1_pos_mask_lr]
        # neg_a3b1_lr = neg_a3b1[a3b1_neg_mask_lr]
        stored_data[a_i] = {"pos_data": pos_ab_data, "neg_data": neg_ab_data}
        # # above 3 all smaller than 0.5 right to left direction)
        # a3b1_pos_mask_rl = (pos_a3b1[:, :, -1] < 0.5).sum(dim=-1) == 5
        # a3b1_neg_mask_rl = (neg_a3b1[:, :, -1] < 0.5).sum(dim=-1) == 5
        # pos_a3b1_rl = pos_a3b1[a3b1_pos_mask_rl]
        # neg_a3b1_rl = neg_a3b1[a3b1_neg_mask_rl]
        # stored_data[a_i]["a3b1_rl"] = {"pos_data": pos_a3b1_rl, "neg_data": neg_a3b1_rl}
        #
        # # existence of above 3 and below 0
        # a3b0_pos_mask =(pos_ab_data[:, -1, 1] == 0) & (pos_ab_data[:, 1:, 1].sum(dim=-1) == 3)
        # a3b0_neg_mask = (neg_ab_data[:, -1, 1] == 0) & (neg_ab_data[:, 1:, 1].sum(dim=-1) == 3)
        # pos_a3b0 = pos_ab_data[a3b0_pos_mask]
        # neg_a3b0 = neg_ab_data[a3b0_neg_mask]
        # # above 3 all greater than 0.5 (left to right direction)
        # a3b0_pos_mask_lr = (pos_a3b0[:, :, -1] > 0.5).sum(dim=-1) == 4
        # a3b0_neg_mask_lr = (neg_a3b0[:, :, -1] > 0.5).sum(dim=-1) == 4
        # pos_a3b0_lr = pos_a3b0[a3b0_pos_mask_lr]
        # neg_a3b0_lr = neg_a3b0[a3b0_neg_mask_lr]
        # stored_data[a_i]["a3b0_lr"] = {"pos_data": pos_a3b0_lr, "neg_data": neg_a3b0_lr}
        #
        # # above 3 all smaller than 0.5 right to left direction)
        # a3b0_pos_mask_rl = (pos_a3b0[:, :, -1] < 0.5).sum(dim=-1) == 4
        # a3b0_neg_mask_rl = (neg_a3b0[:, :, -1] < 0.5).sum(dim=-1) == 4
        # pos_a3b0_rl = pos_a3b0[a3b0_pos_mask_rl]
        # neg_a3b0_rl = neg_a3b0[a3b0_neg_mask_rl]
        # stored_data[a_i]["a3b0_rl"] = {"pos_data": pos_a3b0_rl, "neg_data": neg_a3b0_rl}

    torch.save(stored_data, data_file)
    print(f"Saved data to {data_file}.")
