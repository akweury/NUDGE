# Created by jing at 18.01.24
import random

import torch
from torch import nn as nn
from torch.distributions import Categorical

from pi import Reasoner
from pi.utils import oc_utils


class SymbolicMicroProgramModel(nn.Module):
    def __init__(self, args, rng=None):
        super(SymbolicMicroProgramModel, self).__init__()
        self.rng = random.Random() if rng is None else rng
        self.args = args
        self.actor = Reasoner.SmpReasoner(args)

    def forward(self):
        raise NotImplementedError

    def act(self, logic_state, epsilon=0.0):
        action_probs = self.actor(logic_state)

        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
            action = dist.sample()
        else:
            dist = Categorical(action_probs)
            action = (action_probs[0] == max(action_probs[0])).nonzero(as_tuple=True)[0].squeeze(0).to(self.args.device)
            if torch.numel(action) > 1:
                action = action[0]
        # action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, neural_state, logic_state, action):
        action_probs = self.actor(logic_state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(neural_state)

        return action_logprobs, state_values, dist_entropy

    def get_prednames(self):
        return self.actor.get_prednames()


class SymbolicMicroProgramPlayer:
    def __init__(self, args):
        self.args = args
        self.preds = None
        self.model = SymbolicMicroProgramModel(args).actor.to(args.device)

    def update(self, args=None, behaviors=None, prop_indices=None, explains=None, preds=None):
        if args is not None:
            self.args = args
        if preds is not None:
            self.preds = preds
        self.model.update(args, behaviors, prop_indices, explains, preds)

    def act(self, state):
        if self.args.m == 'getout' or self.args.m == "getoutplus":
            action, explaining = self.getout_actor(state)
        elif self.args.m == 'Assault':
            action, explaining = self.assault_actor(state)
        elif self.args.m == 'threefish':
            action, explaining = self.threefish_actor(state)
        elif self.args.m == 'loot':
            action, explaining = self.loot_actor(state)
        elif self.args.m == 'atari':
            action, explaining = self.atari_actor(state)
        else:
            raise ValueError
        return action, explaining

    def reasoning_act(self, state):
        if self.args.m == 'getout':
            action, explaining = self.getout_reasoning_actor(state)
        else:
            raise ValueError
        return action, explaining

    def get_probs(self):
        probs = self.model.get_probs()
        return probs

    def get_explaining(self):
        explaining = 0
        return explaining

    def get_state(self, state):
        if self.args.m == 'Assault':
            logic_state = oc_utils.extract_logic_state_assault(state, self.args).squeeze(0)
        else:
            raise ValueError
        logic_state = logic_state.tolist()
        result = []
        for list in logic_state:
            obj_state = [round(num, 2) for num in list]
            result.append(obj_state)
        return result

    def getout_actor(self, getout):
        extracted_state = oc_utils.extract_logic_state_getout(getout, self.args)
        predictions, explains = self.model(extracted_state)
        if predictions.sum() > 1:
            print("watch")
        prediction = torch.argmax(predictions).cpu().item()
        # explaining = explains[prediction]
        explaining = None
        action = prediction + 1
        return action, explaining

    def action_combine_assault(self, action_prob, explains):

        if explains == [-1]:
            return action_prob, explains

        if action_prob.sum() <= 1:
            explains = [b_i for action_prob, b_i in explains]
            return action_prob, explains

        if action_prob[0, 0] == 1:
            action_prob[0, 0] = 0
            explains = [(action_prob, b_i) for action_prob, b_i in explains if not action_prob[0] == 1]

        if action_prob[0, 1] == action_prob[0, 2] == 1:
            action_prob[0, 2] = 0
            explains = [(action_prob, b_i) for action_prob, b_i in explains if not action_prob[2] == 1]

        # left and right
        if action_prob[0, 3] == action_prob[0, 4] == 1:
            action_prob[0, 3] = 0
            action_prob[0, 4] = 0
            explains = [(action_prob, b_i) for action_prob, b_i in explains if
                        not (action_prob[3] == 1 or action_prob[4] == 1)]

        # left fire and right fire
        if action_prob[0, 5] == action_prob[0, 6] == 1:
            action_prob[0, 5] = 0
            explains = [(action_prob, b_i) for action_prob, b_i in explains if not (action_prob[5] == 1)]

        if sum(action_prob[0, [5, 6]]) > 0 and action_prob[0, 1] == 1:
            action_prob[0, 1] = 0
            explains = [(action_prob, b_i) for action_prob, b_i in explains if not (action_prob[1] == 1)]

        if action_prob.sum() > 1:
            raise ValueError
        explains = [b_i for action_prob, b_i in explains]
        return action_prob, explains

    def assault_actor(self, getout):
        extracted_state = oc_utils.extract_logic_state_assault(getout, self.args).unsqueeze(0)
        predictions, explains = self.model(extracted_state)
        predictions, explains = self.action_combine_assault(predictions, explains)
        prediction = torch.argmax(predictions).cpu().item()
        # explaining = explains[prediction]
        explaining = None
        action = prediction
        return action, explaining

    def getout_reasoning_actor(self, getout):
        extracted_state = extract_logic_state_getout(getout, self.args)
        predictions, explain = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()

        action = prediction + 1
        return action, explain

    def atari_actor(self, atari_env):
        # import ipdb; ipdb.set_trace()
        extracted_state = extract_logic_state_atari(atari_env, self.args)
        predictions = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()
        explaining = self.prednames[prediction]
        action = preds_to_action_atari(prediction, self.prednames)
        return action, explaining

    def threefish_actor(self, state):
        state = extract_logic_state_threefish(state, self.args)
        predictions = self.model(state)
        action = torch.argmax(predictions)
        explaining = self.prednames[action.item()]
        action = preds_to_action_threefish(action, self.prednames)
        return action, explaining

    def loot_actor(self, state):
        state = extract_logic_state_loot(state, self.args)
        predictions = self.model(state)
        action = torch.argmax(predictions)
        explaining = self.prednames[action.item()]
        action = preds_to_action_loot(action, self.prednames)
        return action, explaining
