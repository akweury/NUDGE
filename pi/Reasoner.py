# Created by jing at 01.12.23

import torch
import torch.nn as nn

from pi.MicroProgram import MicroProgram, UngroundedMicroProgram


class SmpReasoner(nn.Module):
    """ take game state as input, produce conf. of each actions
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.smps = None
        self.ungrounded_smps = None
        self.obj_type_indices = None
        self.explains = None

        # self.action_prob = torch.zeros(len(args.action_names)).to(args.device)

    def update(self, smps, ungrounded_smps, obj_type_indices, explains):
        self.smps = smps
        self.ungrounded_smps = ungrounded_smps
        self.obj_type_indices = obj_type_indices
        self.explains = explains

    def action_combine(self, action_prob):

        if action_prob.sum() <= 1:
            return action_prob

        # left canceled out right
        if action_prob[0, 0] == action_prob[0, 1] == 1:
            action_prob[0, 0] = 0
            action_prob[0, 1] = 0
        elif action_prob[0, 0] == action_prob[0, 2] == 1:
            action_prob[0, 0] = 0
        elif action_prob[0, 1] == action_prob[0, 2] == 1:
            action_prob[0, 1] = 0
        else:
            raise ValueError
        return action_prob

    def grounding(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        # only return switch of actions,
        #       action equals to 1 <==> action is passed for this smp
        #       action equals to 0 <==> action is not passed for this smp
        # calculate the switches of each action
        action_prob = torch.zeros(1, len(self.args.action_names)).to(self.args.device)
        ungrounded_smp_dicts = []
        explains = ["unknown"] * len(self.args.action_names)

        # taking a random action
        if self.smps is None or len(self.smps) == 0:
            action_prob = torch.zeros(1, len(self.args.action_names))
            action_prob[0, torch.randint(0, len(self.args.action_names), (1,))] = 1
            explains = ["random_decision"] * len(self.args.action_names)
        else:
            for smp in self.smps:
                action_probs, params = smp(x, self.obj_type_indices)
                action_prob += action_probs
                explains = self.explains

        if self.ungrounded_smps is not None:
            for ungrounded_smp in self.ungrounded_smps:
                ungrounded_action_prob, params = ungrounded_smp(x, self.obj_type_indices)
                valid_index = ungrounded_action_prob.sum(dim=2).squeeze() > 0
                ungrounded_action = torch.argmax(ungrounded_action_prob, dim=2).squeeze() + 1
                ungrounded_smp_dict = {"valid": valid_index,
                                       "reward": ungrounded_smp.expected_reward,
                                       "action": ungrounded_action}
                ungrounded_smp_dicts.append(ungrounded_smp_dict)

        action_prob = action_prob / (action_prob + 1e-20)
        action_prob = self.action_combine(action_prob)

        return action_prob, ungrounded_smp_dicts, explains

    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        # only return switch of actions,
        #       action equals to 1 <==> action is passed for this smp
        #       action equals to 0 <==> action is not passed for this smp
        # calculate the switches of each action
        action_prob = torch.zeros(1, len(self.args.action_names)).to(self.args.device)
        ungrounded_action_probs = []
        explains = ["unknown"] * len(self.args.action_names)

        # taking a random action
        if self.smps is None or len(self.smps) == 0:
            action_prob = torch.zeros(1, len(self.args.action_names))
            action_prob[0, torch.randint(0, len(self.args.action_names), (1,))] = 1
            explains = ["random_decision"] * len(self.args.action_names)
        else:
            for smp in self.smps:
                action_probs, params = smp(x, self.obj_type_indices)
                action_prob += action_probs
                explains = self.explains

        if self.ungrounded_smps is not None:
            for ungrounded_smp in self.ungrounded_smps:
                ungrounded_action_prob, params = ungrounded_smp(x, self.obj_type_indices)
                ungrounded_action_probs.append(ungrounded_action_prob)

        action_prob = action_prob / (action_prob + 1e-20)
        action_prob = self.action_combine(action_prob)

        return action_prob, explains
