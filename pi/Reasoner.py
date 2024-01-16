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

    def update(self, args, behaviors, game_info, prop_indices, explains, preds):
        if args is not None:
            self.args = args
        if behaviors is not None:
            self.behaviors = behaviors
        if game_info is not None:
            self.game_info = game_info
        if prop_indices is not None:
            self.prop_indices = prop_indices
        if explains is not None:
            self.explains = explains
        if preds is not None:
            self.preds = preds

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

    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        action_prob = torch.zeros(1, len(self.args.action_names)).to(self.args.device)
        explains = "unknown"

        # taking a random action
        if self.behaviors is None or len(self.behaviors) == 0:
            action_prob = torch.zeros(1, len(self.args.action_names))
            action_prob[0, torch.randint(0, len(self.args.action_names), (1,))] = 1
            explains = "random"
        else:
            for behavior in self.behaviors:
                satisfaction, action_probs = behavior.eval_behavior(self.preds, x, self.game_info)
                if satisfaction:
                    action_prob += action_probs
                    explains = self.explains

        action_prob = action_prob / (action_prob + 1e-20)
        action_prob = self.action_combine(action_prob)

        return action_prob, explains
