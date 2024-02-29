# Created by jing at 01.12.23

import torch
import torch.nn as nn
from pi.utils import math_utils


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

    def set_model(self):
        # convert behaviors to tensors

        self.o = torch.zeros(len(self.behaviors), 2, dtype=torch.int).to(self.args.device)
        self.o_mask = torch.zeros(len(self.behaviors), len(self.args.obj_info), dtype=torch.bool).to(
            self.args.device)
        self.p = torch.zeros(len(self.behaviors), 2, dtype=torch.int).to(self.args.device)
        self.actions = torch.zeros(len(self.behaviors), 1, dtype=torch.int).to(self.args.device)
        self.move_directions = torch.zeros(len(self.behaviors)).to(self.args.device)
        self.dir_types = torch.zeros(len(self.behaviors)).to(self.args.device)
        self.x_types = torch.zeros(len(self.behaviors)).to(self.args.device)
        self.y_types = torch.zeros(len(self.behaviors)).to(self.args.device)
        self.mask_pos_att_beh = torch.zeros(len(self.behaviors), dtype=torch.bool).to(self.args.device)
        self.mask_neg_beh = torch.zeros(len(self.behaviors), dtype=torch.bool).to(self.args.device)
        self.mask_pos_beh = torch.zeros(len(self.behaviors), dtype=torch.bool).to(self.args.device)
        self.beh_skill_stage = torch.zeros(len(self.behaviors)).to(self.args.device)
        self.beh_weights = torch.zeros(len(self.behaviors)).to(self.args.device)
        self.o2_indices = torch.zeros(len(self.behaviors), self.args.game_info["state_row_num"], dtype=torch.bool).to(
            self.args.device)
        for b_i, beh in enumerate(self.behaviors):
            self.o[b_i] = torch.tensor(beh.fact[0].obj_comb)
            self.o_mask[b_i, beh.fact[0].obj_comb] = True
            obj_1_indices = self.args.obj_info[self.o[b_i, 1]]["indices"]
            self.o2_indices[b_i, obj_1_indices] = True
            self.beh_weights[b_i] = beh.reward
            self.p[b_i] = torch.tensor(beh.fact[0].prop_comb)
            self.dir_types[b_i] = beh.fact[0].preds[0].dir_type
            self.x_types[b_i] = beh.fact[0].preds[0].x_type
            self.y_types[b_i] = beh.fact[0].preds[0].y_type
            self.actions[b_i] = beh.action

            self.move_directions[b_i] = math_utils.action_to_deg(self.args.action_names[beh.action])
            if beh.skill_beh:
                self.mask_pos_att_beh[b_i] = True
            elif beh.neg_beh:
                self.mask_neg_beh[b_i] = True
            else:
                self.mask_pos_beh[b_i] = True
        self.beh_weights = (self.beh_weights - self.beh_weights.min()) / (
                    self.beh_weights.max() - self.beh_weights.min())

    def update(self, args, behaviors):
        if args is not None:
            self.args = args
        if behaviors is not None:
            self.behaviors = behaviors
            self.set_model()

    def eval_behaviors(self, x):

        # evaluate state
        xs = torch.repeat_interleave(x, len(self.behaviors), 0)
        o1 = xs[:, 0:1]
        o2 = xs[:, 1:]

        p1 = torch.cat((o1[torch.arange(xs.shape[0]), :, self.p[:, 0]].unsqueeze(-1),
                        o1[torch.arange(xs.shape[0]), :, self.p[:, 1]].unsqueeze(-1)), dim=-1)
        p2 = torch.cat((o2[torch.arange(xs.shape[0]), :, self.p[:, 0]].unsqueeze(-1),
                        o2[torch.arange(xs.shape[0]), :, self.p[:, 1]].unsqueeze(-1)), dim=-1)

        p1_moved = math_utils.one_step_move_o2o(p1, self.move_directions, self.args.step_dist)
        dists = math_utils.dist_a_and_b(p1_moved, p2)
        dists = math_utils.closest_one_percent(dists, 0.05)
        dirs = math_utils.dir_a_and_b_with_alignment(p1_moved, p2).reshape(-1, p2.shape[1])

        mask_dir_eq = torch.eq(dirs, self.dir_types.unsqueeze(1))
        mask_x_eq = torch.eq(dists[:, :, 0], self.x_types.unsqueeze(1))
        mask_y_eq = torch.eq(dists[:, :, 1], self.y_types.unsqueeze(1))
        mask = mask_dir_eq * mask_x_eq * mask_y_eq
        mask[~self.o_mask[:, 1:]] = False
        mask_sum = mask.sum(dim=1).to(torch.bool)
        self.beh_skill_stage[~mask_sum] = 0
        conf= mask_sum.float() * self.beh_weights
        return conf

    def get_beh_mask(self, x):

        mask_pos_beh = torch.zeros(len(self.behaviors), dtype=torch.bool)
        mask_neg_beh = torch.zeros(len(self.behaviors), dtype=torch.bool)
        mask_pos_att_beh = torch.zeros(len(self.behaviors), dtype=torch.bool)
        # predictions = torch.zeros(len(self.behaviors))
        confidences = torch.zeros(len(self.behaviors))
        for b_i, beh in enumerate(self.behaviors):

            confidences[b_i] = beh.eval_o2o_behavior(x, self.args.obj_info) * beh.weight
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
        beh_conf = self.eval_behaviors(x) * self.weights
        # mask_pos_beh, mask_neg_beh, mask_pos_att_beh, beh_confidence = self.get_beh_mask(x)

        mask_neg_beh = self.mask_neg_beh * (beh_conf > 0)
        mask_pos_beh = self.mask_pos_beh * (beh_conf > 0)
        mask_pos_att_beh = self.mask_pos_att_beh * (beh_conf > 0)
        # defense behavior
        if mask_neg_beh.sum() > 0:
            beh_neg_indices = torch.arange(len(self.behaviors))[mask_neg_beh]
            for neg_index in beh_neg_indices:
                if beh_conf[neg_index] > 0:
                    beh = self.behaviors[neg_index]
                    pred_action = beh.action
                    action_mask[0, pred_action] = True
                    explains["behavior_index"].append(neg_index)
                    explains["behavior_action"].append(beh.action)
                    explains["behavior_conf"].append(beh_conf[neg_index])
        # skill behavior
        has_skill_beh = False
        if mask_pos_att_beh.sum() > 0:
            beh_skill_pos_indices = torch.arange(len(self.behaviors))[mask_pos_att_beh]
            for pos_skill_index in beh_skill_pos_indices:
                beh = self.behaviors[pos_skill_index]
                use_beh = False
                if not action_mask[0, beh.action[beh.skill_stage]]:
                    if beh_conf[pos_skill_index] > 0:
                        if beh.beh_type == "skill_attack":
                            action_prob[0, beh.action] = beh_conf[pos_skill_index]
                            explains["behavior_index"].append(pos_skill_index)
                            explains["behavior_conf"].append(beh_conf[pos_skill_index])
                            explains["behavior_action"].append(beh.action[beh.skill_stage])
                            has_skill_beh = True
                            use_beh = True
                if not use_beh:
                    beh.skill_stage = 0
        # attack behavior
        has_att_beh = False
        if not has_skill_beh and mask_pos_beh.sum() > 0:
            beh_pos_indices = torch.arange(len(self.behaviors)).to(mask_pos_beh.device)[mask_pos_beh]
            for pos_index in beh_pos_indices:
                beh = self.behaviors[pos_index]
                if not action_mask[0, beh.action]:
                    if beh_conf[pos_index] > 0:
                        if beh.beh_type == "attack":
                            action_prob[0, beh.action] = beh_conf[pos_index]
                            explains["behavior_index"].append(pos_index)
                            explains["behavior_conf"].append(beh_conf[pos_index])
                            explains["behavior_action"].append(beh.action)
                            has_att_beh = True

        # path finding behavior
        if not has_att_beh and not has_skill_beh and mask_pos_beh.sum() > 0:
            beh_pos_indices = torch.arange(len(self.behaviors)).to(mask_pos_beh.device)[mask_pos_beh]
            for pos_index in beh_pos_indices:
                beh = self.behaviors[pos_index]
                if not action_mask[0, beh.action]:
                    if beh_conf[pos_index] > 0.1:
                        if beh.beh_type == "path_finding" or beh.beh_type == "o2o":
                            action_prob[0, beh.action] = beh_conf[pos_index]
                            explains["behavior_index"].append(pos_index)
                            explains["behavior_conf"].append(beh_conf[pos_index])
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
