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

    def update(self, args, behaviors):
        if args is not None:
            self.args = args
        if behaviors is not None:
            self.behaviors = behaviors

    def get_beh_mask(self, x):

        mask_pos_beh = torch.zeros(len(self.behaviors), dtype=torch.bool)
        mask_neg_beh = torch.zeros(len(self.behaviors), dtype=torch.bool)
        # predictions = torch.zeros(len(self.behaviors))
        confidences = torch.zeros(len(self.behaviors))
        for b_i, beh in enumerate(self.behaviors):

            confidences[b_i] = beh.eval_behavior(x, self.args.obj_info)
            if beh.neg_beh:
                mask_neg_beh[b_i] = True
            else:
                mask_pos_beh[b_i] = True

        return mask_pos_beh, mask_neg_beh, confidences

    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        action_prob = torch.zeros(1, len(self.args.action_names))
        action_mask = torch.zeros(1, len(self.args.action_names), dtype=torch.bool)
        explains = {"behavior_index": [], "reward": [], 'state':x}
        mask_pos_beh, mask_neg_beh, beh_confidence = self.get_beh_mask(x)

        mask_neg_beh = mask_neg_beh * (beh_confidence > 0)
        mask_pos_beh = mask_pos_beh * (beh_confidence > 0)

        # defense behavior
        if mask_neg_beh.sum() > 0:
            beh_neg_indices = torch.arange(len(self.behaviors))[mask_neg_beh]
            for neg_index in beh_neg_indices:
                beh = self.behaviors[neg_index]
                pred_action = beh.action
                action_mask[0, pred_action] = True
                explains["behavior_index"].append(neg_index)

        # path finding behavior

        if mask_pos_beh.sum() > 0:
            for beh_index in mask_pos_beh.nonzero().reshape(-1):
                behavior = self.behaviors[beh_index]
                if not action_mask[0, behavior.action]:
                    action_prob[0, behavior.action] = beh_confidence[beh_index]
                    explains["behavior_index"].append(beh_index)

        action_prob[action_mask] = -1e+20
        return action_prob, explains
