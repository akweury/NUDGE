# Created by jing at 01.12.23

import torch
import torch.nn as nn


class SmpReasoner(nn.Module):
    """ take game state as input, produce conf. of each actions
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.smps = None
        self.obj_type_indices = None
        self.action_prob = None

        # self.action_prob = torch.zeros(len(args.action_names)).to(args.device)

    def update(self, smps, obj_type_indices):
        self.smps = smps
        self.obj_type_indices = obj_type_indices

    def action_combine(self):

        if self.action_prob.sum() <= 1:
            return

        # left canceled out right
        if self.action_prob[0, 0] == self.action_prob[0, 1] == 1:
            self.action_prob[0, 0] = 0
            self.action_prob[0, 1] = 0
        elif self.action_prob[0, 0] == self.action_prob[0, 2] == 1:
            self.action_prob[0, 0] = 0
        elif self.action_prob[0, 1] == self.action_prob[0, 2] == 1:
            self.action_prob[0, 1] = 0
        else:
            raise ValueError

    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        # only return switch of actions,
        #       action equals to 1 <==> action is passed for this smp
        #       action equals to 0 <==> action is not passed for this smp
        # calculate the switches of each action
        self.action_prob = torch.zeros(1, len(self.args.action_names)).to(self.args.device)
        for smp in self.smps:
            action_probs, params = smp(x, self.obj_type_indices)
            self.action_prob += action_probs

        self.action_prob = self.action_prob / (self.action_prob + 1e-20)
        self.action_combine()

        return self.action_prob
