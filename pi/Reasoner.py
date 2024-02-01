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
            scores = torch.zeros(len(self.behaviors))
            satisfactions = torch.zeros(len(self.behaviors), dtype=torch.bool)
            actions = torch.zeros(len(self.behaviors))

            for b_i, beh in enumerate(self.behaviors):
                satisfactions[b_i], scores[b_i] = beh.eval_behavior(x, self.args.obj_info)
                if satisfactions[b_i] and beh.neg_beh:
                    action_mask[0, beh.action] = False
                    scores[b_i] = 0
                actions[b_i] = beh.action


            passed_scores = scores[satisfactions]
            passed_actions = actions[satisfactions]
            if len(passed_scores)>0:
                beh_index = passed_scores.argmax()
                pred_action = passed_actions[beh_index].int()
                action_prob[0, pred_action] = 1
                explains.append((pred_action, beh_index))
        action_prob = action_prob / (action_prob + 1e-20)
        action_prob[~action_mask] = -1
        print(f"action prob: {action_prob}")
        # if no action was predicted
        if torch.abs(action_prob).sum() == 0:
            explains = [-1]
            action_prob = torch.zeros(1, len(self.args.action_names))
            action_prob[0, torch.randint(0, len(self.args.action_names), (1,))] = 1

        return action_prob, explains
