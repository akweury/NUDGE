# Created by jing at 01.12.23

import torch
import os
import torch.nn as nn

from pi import micro_program_generator


class SmpReasoner(nn.Module):
    """ take game state as input, produce conf. of each actions
    """

    def __init__(self, args, clauses):
        super().__init__()
        self.args = args
        self.smps = micro_program_generator.clauses2smps(args, clauses)
        self.action_switches = torch.zeros(len(self.action_names)).to(args.device)
        self.action_weights = torch.zeros(len(self.action_names)).to(args.device)
    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        # only return switch of actions,
        #       action equals to 1 <==> action is passed for this smp
        #       action equals to 0 <==> action is not passed for this smp

        # Calculate the switches of each action
        for smp in self.smps:
            action_index = smp(x)
            if action_index is not None:
                self.action_switches[action_index] = 1

        # calculate the weight of each action



        return self.action_switches
