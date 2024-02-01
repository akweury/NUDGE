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
        action_mask = torch.ones(1, len(self.args.action_names), dtype=torch.bool)
        # taking a random action
        if self.behaviors is None or len(self.behaviors) == 0:
            action_prob = torch.zeros(1, len(self.args.action_names))
            action_prob[0, torch.randint(0, len(self.args.action_names), (1,))] = 1
            explains = "random"
        else:
            explains = []

            for b_i, beh in enumerate(self.behaviors):
                satisfaction = beh.eval_behavior(x, self.args.obj_info)
                if satisfaction:
                    if beh.neg_beh:
                        action_mask[0, beh.action] = False
                    else:
                        action_prob[0, beh.action] = 1
                    explains.append((beh.action, b_i))
                    print(f"pass behavior: {beh.clause}")
        action_prob = action_prob / (action_prob + 1e-20)
        action_prob[~action_mask] = -1
        print(f"action prob: {action_prob}")
        # if no action was predicted
        if torch.abs(action_prob).sum() == 0:
            explains = [-1]
            action_prob = torch.zeros(1, len(self.args.action_names))
            action_prob[0, torch.randint(0, len(self.args.action_names), (1,))] = 1

        return action_prob, explains
