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
        mask_pos_att_beh = torch.zeros(len(self.behaviors), dtype=torch.bool)
        # predictions = torch.zeros(len(self.behaviors))
        confidences = torch.zeros(len(self.behaviors))
        for b_i, beh in enumerate(self.behaviors):

            confidences[b_i] = beh.eval_behavior(x, self.args.obj_info)
            if beh.skill_beh:
                mask_pos_att_beh[b_i] = True
                # reset skill stage if it is disabled
                if confidences[b_i] == 0:
                    beh.skill_stage = 0
            elif beh.neg_beh:
                mask_neg_beh[b_i] = True
            else:
                mask_pos_beh[b_i] = True

        return mask_pos_beh, mask_neg_beh, mask_pos_att_beh, confidences

    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        action_prob = torch.zeros(1, len(self.args.action_names))
        action_mask = torch.zeros(1, len(self.args.action_names), dtype=torch.bool)
        explains = {"behavior_index": [], "reward": [], 'state': x, "behavior_conf": [], "behavior_action": []}
        mask_pos_beh, mask_neg_beh, mask_pos_att_beh, beh_confidence = self.get_beh_mask(x)

        mask_neg_beh = mask_neg_beh * (beh_confidence > 0)
        mask_pos_beh = mask_pos_beh * (beh_confidence > 0)
        mask_pos_att_beh = mask_pos_att_beh * (beh_confidence > 0)
        # defense behavior
        if mask_neg_beh.sum() > 0:
            beh_neg_indices = torch.arange(len(self.behaviors))[mask_neg_beh]
            for neg_index in beh_neg_indices:
                if beh_confidence[neg_index] > 0:
                    beh = self.behaviors[neg_index]
                    pred_action = beh.action
                    action_mask[0, pred_action] = True
                    explains["behavior_index"].append(neg_index)
                    explains["behavior_action"].append(beh.action)
                    explains["behavior_conf"].append(beh_confidence[neg_index])
        # skill behavior
        has_skill_beh = False
        if mask_pos_att_beh.sum() > 0:
            beh_skill_pos_indices = torch.arange(len(self.behaviors))[mask_pos_att_beh]
            for pos_skill_index in beh_skill_pos_indices:
                beh = self.behaviors[pos_skill_index]
                use_beh = False
                if not action_mask[0, beh.action[beh.skill_stage]]:
                    if beh_confidence[pos_skill_index] > 0:
                        if beh.beh_type == "skill_attack":
                            action_prob[0, beh.action] = beh_confidence[pos_skill_index]
                            explains["behavior_index"].append(pos_skill_index)
                            explains["behavior_conf"].append(beh_confidence[pos_skill_index])
                            explains["behavior_action"].append(beh.action[beh.skill_stage])
                            has_skill_beh = True
                            use_beh = True
                if not use_beh:
                    beh.skill_stage = 0
        # attack behavior
        has_att_beh = False
        if not has_skill_beh and mask_pos_beh.sum() > 0:
            beh_pos_indices = torch.arange(len(self.behaviors))[mask_pos_beh]
            for pos_index in beh_pos_indices:
                beh = self.behaviors[pos_index]
                if not action_mask[0, beh.action]:
                    if beh_confidence[pos_index] > 0:
                        if beh.beh_type == "attack":
                            action_prob[0, beh.action] = beh_confidence[pos_index]
                            explains["behavior_index"].append(pos_index)
                            explains["behavior_conf"].append(beh_confidence[pos_index])
                            explains["behavior_action"].append(beh.action)
                            has_att_beh = True

        # path finding behavior
        if not has_att_beh and not has_skill_beh and mask_pos_beh.sum() > 0:
            beh_pos_indices = torch.arange(len(self.behaviors))[mask_pos_beh]
            for pos_index in beh_pos_indices:
                beh = self.behaviors[pos_index]
                if not action_mask[0, beh.action]:
                    if beh_confidence[pos_index] > 0.1:
                        if beh.beh_type == "path_finding":
                            action_prob[0, beh.action] = beh_confidence[pos_index]
                            explains["behavior_index"].append(pos_index)
                            explains["behavior_conf"].append(beh_confidence[pos_index])
                            explains["behavior_action"].append(beh.action)

        # update skill stage of a skill behavior
        action_prob[action_mask] = 0
        if len(explains["behavior_index"]) > 0:
            optimal_action = action_prob.argmax()
            for beh_i in explains["behavior_index"]:
                beh = self.behaviors[beh_i]
                if beh.skill_beh:
                    if beh.action[beh.skill_stage] == optimal_action:
                        beh.skill_stage += 1
                        beh.skill_stage %= len(beh.action)
                    else:
                        beh.skill_stage = 0


        return action_prob, explains
