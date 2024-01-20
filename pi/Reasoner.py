# Created by jing at 01.12.23

import torch
import torch.nn as nn
from pi.utils import smp_utils


class SmpReasoner(nn.Module):
    """ take game state as input, produce conf. of each actions
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.behaviors = None
        self.obj_type_indices = None
        self.explains = None

        # self.action_prob = torch.zeros(len(args.action_names)).to(args.device)

    def update(self, args, behaviors, prop_indices, explains, preds):
        if args is not None:
            self.args = args
        if behaviors is not None:
            self.behaviors = behaviors
        if prop_indices is not None:
            self.prop_indices = prop_indices
        if explains is not None:
            self.explains = explains
        if preds is not None:
            self.preds = preds

    def action_combine(self, action_prob, explains):

        if action_prob.sum() <= 1:
            explains = [b_i for action_prob, b_i in explains]
            return action_prob, explains

        # left canceled out right
        if action_prob[0, 0] == action_prob[0, 1] == 1:
            action_prob[0, 0] = 0
            action_prob[0, 1] = 0
            explains = [b_i for action_prob, b_i in explains if not (action_prob[0] == 1 or action_prob[1] == 1)]
        elif action_prob[0, 0] == action_prob[0, 2] == 1:
            action_prob[0, 0] = 0
            explains = [b_i for action_prob, b_i in explains if not (action_prob[0] == 1)]
        elif action_prob[0, 1] == action_prob[0, 2] == 1:
            action_prob[0, 1] = 0
            explains = [b_i for action_prob, b_i in explains if not (action_prob[1] == 1)]
        else:
            raise ValueError
        return action_prob, explains

    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        action_prob = torch.zeros(1, len(self.args.action_names)).to(self.args.device)
        explains = "unknown"
        print(x)
        # taking a random action
        if self.behaviors is None or len(self.behaviors) == 0:
            action_prob = torch.zeros(1, len(self.args.action_names))
            action_prob[0, torch.randint(0, len(self.args.action_names), (1,))] = 1
            explains = "random"
        else:
            explains = []
            for b_i in range(len(self.behaviors)):
                satisfaction = self.behaviors[b_i].eval_behavior(self.preds, x, self.args.obj_info)
                if satisfaction:
                    action_prob += self.behaviors[b_i].action
                    explains.append((self.behaviors[b_i].action, b_i))
                    print(f"behavior: {self.behaviors[b_i].clause}")

        action_prob = action_prob / (action_prob + 1e-20)

        if explains is None or len(explains) == 0:
            explains = [-1]
            action_prob = torch.zeros(1, len(self.args.action_names))
            action_prob[0, torch.randint(0, len(self.args.action_names), (1,))] = 1

        return action_prob, explains
