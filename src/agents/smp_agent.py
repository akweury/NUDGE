# Created by jing at 01.12.23


import os
import pickle

import torch
import torch.nn as nn

from .utils_atari import (action_map_atari, extract_logic_state_atari,
                          extract_neural_state_atari)
from .utils_getout import (action_map_getout, extract_logic_state_getout,
                           extract_neural_state_getout)
from .utils_loot import (action_map_loot, extract_logic_state_loot,
                         extract_neural_state_loot)
from .utils_threefish import (action_map_threefish,
                              extract_logic_state_threefish,
                              extract_neural_state_threefish)


class LogicPPO:
    def __init__(self, lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.args = args
        self.policy = NSFR_ActorCritic(self.args).to(device)
        self.optimizer = optimizer([
            {'params': self.policy.actor.get_params(), 'lr': lr_actor},
            {'params': self.policy.critic.p_bound(), 'lr': lr_critic}
        ])

        self.policy_old = NSFR_ActorCritic(self.args).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.prednames = self.get_prednames()

    def select_action(self, state, epsilon=0.0):

        # extract state for different games
        # import ipdb; ipdb.set_trace()
        if self.args.m == 'getout':
            logic_state = extract_logic_state_getout(state, self.args)
            neural_state = extract_neural_state_getout(state, self.args)
        elif self.args.m == 'threefish':
            logic_state = extract_logic_state_threefish(state, self.args)
            neural_state = extract_neural_state_threefish(state, self.args)
        elif self.args.m == 'loot':
            logic_state = extract_logic_state_loot(state, self.args)
            neural_state = extract_neural_state_loot(state, self.args)
        elif self.args.m == 'atari':
            logic_state = extract_logic_state_atari(state, self.args)
            neural_state = extract_neural_state_atari(state, self.args)

        # select random action with epsilon probability and policy probiability with 1-epsilon
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            # import ipdb; ipdb.set_trace()
            action, action_logprob = self.policy_old.act(logic_state, epsilon=epsilon)

        self.buffer.neural_states.append(neural_state)
        self.buffer.logic_states.append(logic_state)
        action = torch.squeeze(action)
        self.buffer.actions.append(action)
        action_logprob = torch.squeeze(action_logprob)
        self.buffer.logprobs.append(action_logprob)

        # different games use different action system, need to map it to the correct action.
        # action of logic game means a String, need to map string to the correct action,
        action = action.item()
        if self.args.m == 'getout':
            action = action_map_getout(action, self.args, self.prednames)
        elif self.args.m == 'threefish':
            action = action_map_threefish(action, self.args, self.prednames)
        elif self.args.m == 'loot':
            action = action_map_loot(action, self.args, self.prednames)
        elif self.args.m == 'atari':
            action = action_map_atari(action, self.args, self.prednames)

        return action

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor

        old_neural_states = torch.squeeze(torch.stack(self.buffer.neural_states, dim=0)).detach().to(device)
        old_logic_states = torch.squeeze(torch.stack(self.buffer.logic_states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_neural_states, old_logic_states,
                                                                        old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # training does not converge if the entropy term is added ...
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)  # - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # wandb.log({"loss": loss})

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path, directory, step_list, reward_list, weight_list):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        with open(directory + '/' + "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)
            pickle.dump(weight_list, f)

    def load(self, directory):
        # only for recover form crash
        model_name = input('Enter file name: ')
        model_file = os.path.join(directory, model_name)
        self.policy_old.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        with open(directory + '/' + "data.pkl", "rb") as f:
            step_list = pickle.load(f)
            reward_list = pickle.load(f)
            weight_list = pickle.load(f)
        return step_list, reward_list, weight_list

    def get_predictions(self, state):
        self.prediction = state
        return self.prediction

    def get_weights(self):
        return self.policy.actor.get_params()

    def get_prednames(self):
        return self.policy.actor.get_prednames()


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.neural_states = []
        self.logic_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.neural_states[:]
        del self.logic_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.predictions[:]
