# Created by jing at 01.12.23

import torch
import os
import torch.nn as nn

from pi import smp


class SmpReasoner(nn.Module):
    """ take game state as input, produce conf. of each actions
    """

    def __init__(self, args, clauses):
        super().__init__()
        self.args = args
        self.smps = micro_program_generator.behavior2smps(args, clauses)
        self.action_prob = torch.zeros(len(args.action_names)).to(args.device)

    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        # only return switch of actions,
        #       action equals to 1 <==> action is passed for this smp
        #       action equals to 0 <==> action is not passed for this smp
        self.action_prob = torch.zeros(1, len(self.args.action_names)).to(self.args.device)
        self.action_counter_prob = torch.zeros(1, len(self.args.counter_action_names)).to(self.args.device)
        # calculate the switches of each action
        for smp in self.smps:
            action_probs = smp(x)
            self.action_prob += action_probs

        action_prob = self.action_prob * torch.tensor([[1, 1, 2]]) + self.action_counter_prob
        return action_prob
