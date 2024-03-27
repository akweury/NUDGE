# Created by shaji at 26/03/2024

import os
from tqdm import tqdm
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
from ocatari.core import OCAtari

from pi.utils import draw_utils

BATCH_SIZE = 32
MEMORY_SIZE = 1000000
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.00025
TARGET_UPDATE_FREQ = 5
EPISODES = 1000

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define Q-network
class AtariNet(nn.Module):
    """ Estimator used by DQN-style algorithms for ATARI games.
        Works with DQN, M-DQN and C51.
    """

    def __init__(self, action_no, distributional=False):
        super().__init__()

        self.action_no = out_size = action_no
        self.distributional = distributional

        # configure the support if distributional
        if distributional:
            support = torch.linspace(-10, 10, 51)
            self.__support = nn.Parameter(support, requires_grad=False)
            out_size = action_no * len(self.__support)

        # get the feature extractor and fully connected layers
        self.__features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.__head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Linear(512, out_size),
        )

    def forward(self, x):
        assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div(255)

        x = self.__features(x)
        qs = self.__head(x.view(x.size(0), -1))

        if self.distributional:
            logits = qs.view(qs.shape[0], self.action_no, len(self.__support))
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.__support.expand_as(qs_probs)).sum(2)
        return qs


# Define DQN agent
class DQNAgent:
    def __init__(self, args, input_shape, num_actions):
        self.policy_net = AtariNet(num_actions).to(args.device)
        self.target_net = AtariNet(num_actions).to(args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
        self.num_actions = num_actions

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPSILON_MIN + (EPSILON - EPSILON_MIN) * \
                        np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long).to(state.device)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE).to(non_final_next_states.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + (next_state_values * GAMMA)

        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)  # Output size is 8 for 8 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_nn(args, num_actions, input_tensor, target_tensor, text):
    # Define your neural network architecture

    # Instantiate the model
    model = Classifier(input_tensor.size(1), num_actions).to(input_tensor.device)

    # Define your loss function (cross-entropy for classification)
    criterion = nn.CrossEntropyLoss()

    # Define your optimizer (e.g., SGD, Adam, etc.)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Example data - you should replace this with your actual data
    # Input tensor shape: [batch_size, 16]
    # Target tensor shape: [batch_size]
    # Training loop
    num_epochs = args.train_epochs
    losses = torch.zeros(1, num_epochs)
    for epoch in tqdm(range(num_epochs), desc=f"Train MLP agent {text}"):
        # Forward pass
        outputs = model(input_tensor)

        # Compute loss
        loss = criterion(outputs, target_tensor)

        # Zero gradients, backward pass, and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[0, epoch] = loss.detach()
        # print(f"loss {loss}")

    draw_utils.plot_line_chart(losses, path=args.trained_model_folder, labels=["loss"], title=f"{text}",
                               figure_size=(10, 5))
    return model


def load_mlp_a(model_folder,obj_num, game_name):
    # load MLP-A
    mlp_a = []
    for obj_i in range(obj_num):
        mlp_a_i_file = model_folder / f"{game_name}_mlp_a_{obj_i}.pth.tar"
        mlp_a_i = torch.load(mlp_a_i_file)["model"]
        mlp_a.append(mlp_a_i)
    return mlp_a
