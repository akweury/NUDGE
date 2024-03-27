# Created by jing at 04.12.23

import os

import torch

from pi import train_mlp_a, train_dqn_t, train_mlp_t

train_mlp_a.train_mlp_a()
train_dqn_t.train_dqn_t()
train_mlp_t.train_mlp_t()
