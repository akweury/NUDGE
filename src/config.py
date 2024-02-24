import torch
import math
import os
from pathlib import Path

max_ep_len = 500  # max timesteps in one episode
max_training_timesteps = 800000  # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4  # print avg reward in the interval (in num timesteps)
# print_freq = max_ep_len  # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 4  # log avg reward in the interval (in num timesteps)
save_model_freq = max_ep_len * 50  # save model frequency (in num timesteps)
# save_model_freq = max_ep_len  # save model frequency (in num timesteps)
#####################################################

################ hyperparameters ################

update_timestep = max_ep_len * 2  # update policy every n episodes

K_epochs = 20  # update policy for K epochs (= # update steps)
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

optimizer = torch.optim.Adam
lr_actor = 0.001  # learning rate for actor network
lr_critic = 0.0003  # learning rate for critic network
# epsilon_func = lambda episode: math.exp(-episode / 500)
epsilon_func = lambda episode: max(math.exp(-episode / 500), 0.02)

## paths
root = Path(__file__).parents[0]

path_image = root / "image"
path_runs = root / "runs"
path_model = root / 'models'
path_saved_bs_data = root / "bs_data"

path_output = root / ".." / ".." / "storage"
if not os.path.exists(path_output):
    os.mkdir(path_output)

path_bs_data = path_output / "bs_data"
if not os.path.exists(path_bs_data):
    os.mkdir(path_bs_data)

path_log = path_output / "logs"
if not os.path.exists(path_log):
    os.mkdir(path_log)

path_check_point = path_output / "check_point"
if not os.path.exists(path_check_point):
    os.mkdir(path_check_point)

path_exps = root / "exps"
if not os.path.exists(path_exps):
    os.mkdir(path_exps)

############## settings ###############

mask_splitter = "#"
smp_param_unit = 0.01

########### properties ################

state_idx_getout_x = 4
state_idx_getout_y = 5

state_idx_assault_x = 5
state_idx_assault_y = 6

state_idx_asterix_x = 6
state_idx_asterix_y = 7

state_idx_threefish_agent = 0
state_idx_threefish_fish = 1
state_idx_threefish_radius = 2
state_idx_threefish_x = 3
state_idx_threefish_y = 4

############## object info ##########################

obj_type_name_getout = ['agent', 'key', 'door', 'enemy']

obj_info_getout = [('agent', 1),
                   ('key', 1),
                   ('door', 1),
                   ('enemy', 1)]
game_info_getout = {
    "obj_info": obj_info_getout,
    "state_row_num": 4,
    "state_col_num": 6,
    "axis_x_col": 4,
    "axis_y_col": 5
}

obj_info_getoutplus = [('agent', 1),
                       ('key', 1),
                       ('door', 1),
                       ('enemy', 5)]
game_info_getoutplus = {
    "obj_info": obj_info_getoutplus,
    "state_row_num": 8,
    "state_col_num": 6,
    "axis_x_col": 4,
    "axis_y_col": 5
}
obj_info_assault = [('agent', [0], [0]),
                    ('player_missile_vertical', [1, 2], [1]),
                    ('player_missile_horizontal', [3, 4], [2]),
                    ('enemy', [5, 6, 7, 8, 9], [3]),
                    ('enemy_missile', [10, 11], [4])]

obj_info_asterix = [('Player', 1),
                    ('Enemy', 8),
                    ('Consumable', 8)
                    ]
action_name_asterix = ["noop",  # 0
                       "up",  # 1
                       "right",  # 2
                       "left",  # 3
                       "down",  # 4
                       "upright",  # 5
                       "upleft",  # 6
                       "downright",  # 7
                       "downleft"  # 8
                       ]

game_info_asterix = {
    "obj_info": obj_info_asterix,
    "state_row_num": 17,
    "state_col_num": 5,
    "axis_x_col": 3,
    "axis_y_col": 4
}

obj_info_kangaroo = [('Player', 1),
                     ('Child', 1),
                     ('Fruit', 3),
                     ('Bell', 1),
                     ('Platform', 4),
                     ('Ladder', 3),
                     ('Enemy', 3),
                     ('Projectile_top', 3),
                     ('Projectile_enemy', 3),
                     ]

game_info_kangaroo = {
    "obj_info": obj_info_kangaroo,
    "state_row_num": 22,
    "state_col_num": 11,
    "axis_x_col": 9,
    "axis_y_col": 10
}
obj_info_boxing = [('Player', 1),
                   ('Enemy', 1)]
game_info_boxing = {
    "obj_info": obj_info_boxing,
    "state_row_num": 2,
    "state_col_num": 4,
    "axis_x_col": 2,
    "axis_y_col": 3
}

action_name_boxing = ["noop",  # 0
                      "fire",  # 1
                      "up",  # 2
                      "right",  # 3
                      "left",  # 4
                      "down",  # 5
                      "upright",  # 6
                      "upleft",  # 7
                      "downright",  # 8
                      "downleft",  # 9
                      "upfire",  # 10
                      "rightfire",  # 11
                      "leftfire",  # 12
                      "downfire",  # 13
                      "uprightfire",  # 14
                      "upleftfire",  # 15
                      "downrightfire",  # 16
                      "downleftfire",  # 17
                      ]

########### action info ############################

action_name_getout = ["left", "right", "jump"]
action_name_assault = ["noop",  # 0
                       "fire",  # 1
                       "up",  # 2
                       "right",  # 3
                       "left",  # 4
                       "rightfire",  # 5
                       "leftfire"  # 6
                       ]

action_name_kangaroo = ["noop",  # 0
                        "fire",  # 1
                        "up",  # 2
                        "right",  # 3
                        "left",  # 4
                        "down",  # 5
                        "upright",  # 6
                        "upleft",  # 7
                        "downright",  # 8
                        "downleft",  # 9
                        "upfire",  # 10
                        "rightfire",  # 11
                        "leftfire",  # 12
                        "downfire",  # 13
                        "uprightfire",  # 14
                        "upleftfire",  # 15
                        "downrightfire",  # 16
                        "downleftfire",  # 17
                        ]

action_name_threefish = ["left", "right", "jump"]

################### prop info ########################

prop_name_getout = ['agent', 'key', 'door', 'enemy', "axis_x", "axis_y"]
prop_name_threefish = ['agent', 'fish', "radius", "axis_x", "axis_y"]
prop_name_assault = ['agent', 'player_missile_vertical', "player_missile_horizontal", "enemy", "enemy_missile",
                     "axis_x", "axis_y"]
prop_name_asterix = ["Player", "Enemy", "Consumable", "axis_x", "axis_y"]
prop_name_boxing = ["Player", "Enemy", "axis_x", "axis_y"]
prop_name_kangaroo = ["agent", "monkey", "apple",
                      "axis_x", "axis_y"]
########## language ########################

func_pred_name = "func_pred"
exist_pred_name = "exist_pred"
action_pred_name = "action_pred"
counter_action_pred_name = "counter_action_pred"

################# remove in the future #############

obj_type_indices_getout = {'agent': [0], 'key': [1], 'door': [2], 'enemy': [3]}
obj_type_indices_getout_plus = {'agent': [0], 'key': [1], 'door': [2], 'enemy': [3, 4, 5], 'buzzsaw': [6, 7]}
obj_type_indices_threefish = {'agent': [0], 'fish': [1, 2]}
