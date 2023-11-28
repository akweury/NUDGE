import torch
import math

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
path_check_point =  root /"checkpoints"
path_image = root / "image"
path_runs = root / "runs"
path_model = root / 'models'

action_idx_left = 0
action_idx_right = 1
action_idx_jump = 2

state_idx_player = 0
state_idx_key = 1
state_idx_door = 2
state_idx_enemy = 3
state_idx_x = 4
state_idx_y = 5

state_name_list = ['player', 'key', 'door', 'enemy']