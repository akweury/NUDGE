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

    def get_beh_mask(self, x):

        mask_pos_beh = torch.zeros(len(self.behaviors), dtype=torch.bool)
        mask_neg_beh = torch.zeros(len(self.behaviors), dtype=torch.bool)
        predictions = torch.zeros(len(self.behaviors), 2)
        for b_i, beh in enumerate(self.behaviors):
            predictions[b_i] = beh.eval_behavior(x, self.args.obj_info)
            if beh.neg_beh:
                mask_neg_beh[b_i] = True
            else:
                mask_pos_beh[b_i] = True

        return mask_pos_beh, mask_neg_beh, predictions

    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        action_prob = torch.zeros(1, len(self.args.action_names)).to(self.args.device)
        print(x)
        explains = {"behavior_index": [], "reward": []}
        mask_pos_beh, mask_neg_beh, beh_predictions = self.get_beh_mask(x)
        mask_behs = beh_predictions.argmax(dim=1) == 1
        mask_neg_beh = mask_neg_beh * mask_behs
        mask_pos_beh = mask_pos_beh * mask_behs

        if mask_neg_beh.sum() > 0:
            beh_neg_indices = torch.arange(len(self.behaviors))[mask_neg_beh]
            for neg_index in beh_neg_indices:
                beh = self.behaviors[neg_index]
                pred_action = beh.action
                action_prob[0, pred_action] = -(beh.passed_state_num / (beh.test_passed_state_num + 1e-20))
                explains["behavior_index"].append(neg_index)
                self.get_beh_mask(x)

        if (mask_pos_beh * ~mask_neg_beh).sum() > 0:
            beh_pos_index = beh_predictions[mask_pos_beh * ~mask_neg_beh].argmax()
            best_beh_index = torch.arange(len(self.behaviors))[mask_pos_beh][beh_pos_index]
            pred_action = self.behaviors[best_beh_index].action
            action_prob[0, pred_action] = 1
            explains["behavior_index"].append(best_beh_index)
        return action_prob, explains
