# Created by jing at 04.12.23

import os

import torch

from pi import train_mlp_a, train_dqn_c, train_mlp_c, train_dqn_t

# collect game buffer by dqn-a
# train mlp-a from dqn-a's buffer
dqn_a_avg_score = train_mlp_a.train_mlp_a()

# train dqn-t with using mlp-a
train_dqn_c.train_dqn_c()

# collect game buffer by dqn-t
# train mlp-t from dqn-t's buffer
dqn_c_avg_score = train_mlp_c.train_mlp_c()

# train dqn-r with using mlp-t, mlp-a
train_dqn_t.train_dqn_t()
