# Created by jing at 01.12.23

import torch
import torch.nn as nn
from pi.utils import math_utils, reason_utils


class SmpReasoner(nn.Module):
    """ take game state as input, produce conf. of each actions
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.behaviors = None
        self.obj_type_indices = None
        self.explains = None
        self.last_state = None
        self.last2nd_state = None
        self.predicate_weights_table = None
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

    def update(self, args, behaviors, o2o_data, action_delta):
        self.o2o_data = o2o_data
        self.action_delta = action_delta
        self.o2o_data_weights = torch.ones(len(o2o_data))
        self.game_o2o_weights = []
        self.previous_pred_mask = torch.zeros(len(self.o2o_data), dtype=torch.bool)
        self.pwt = torch.zeros(len(self.o2o_data), len(self.o2o_data))

        if args is not None:
            self.args = args
        if behaviors is not None and len(behaviors) > 0:
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
        conf = mask_sum.float() * self.beh_weights
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

    def eval_o2o_behavior(self, state_tensor):

        for o2o_indices, o2o_beh in self.o2o_data:
            o2o_data = torch.tensor(o2o_beh)[:, :-1]
            o2o_act = torch.tensor(o2o_beh)[:, 2]
            data = state_tensor[o2o_indices]
            mask = (data == o2o_data).prod(dim=1).bool()
            act = o2o_act[mask]
            if len(act) > 0:
                return act[0]
        action = None
        return action

    def get_closest_o2o_beh(self, state_tensor):
        least_dist = 1e+20
        closest_beh = None
        closest_beh_index = None
        for o2o_indices, o2o_beh in self.o2o_data:
            o2o_beh = torch.tensor(o2o_beh)
            o2o_data = o2o_beh[:, :-1]
            data = state_tensor[o2o_indices]
            least_diff_beh_dist = torch.abs(data - o2o_data).sum(dim=1).min()
            index = torch.abs(data - o2o_data).sum(dim=1).argmin()
            if least_diff_beh_dist < least_dist:
                least_dist = least_diff_beh_dist
                closest_beh = o2o_beh[index]
                closest_beh_index = o2o_indices
        return closest_beh, closest_beh_index, least_dist

    def get_next_state_tensor(self, state_time3):

        next_state_tensors = torch.zeros(len(self.action_delta), 5).to(state_time3.device)

        for a_i in range(len(self.action_delta)):
            past_now_next_states = state_time3[a_i]
            past_now_next_states = math_utils.closest_one_percent(past_now_next_states, 0.01)
            obj_ab_dir = math_utils.closest_multiple_of_45(reason_utils.get_ab_dir(past_now_next_states,
                                                                                   0, 1)).reshape(-1)

            obj_velocities = reason_utils.get_state_velo(past_now_next_states)
            obj_velo_dir = math_utils.closest_one_percent(math_utils.get_velo_dir(obj_velocities), 0.01)

            next_state_tensors[a_i, 0] = torch.abs(
                past_now_next_states[-1, 0, -2:-1] - past_now_next_states[-1, 1, -2:-1])
            next_state_tensors[a_i, 1] = torch.abs(past_now_next_states[-1, 0, -1:] - past_now_next_states[-1, 1, -1:])
            next_state_tensors[a_i, 2] = obj_velo_dir[-1, 0]
            next_state_tensors[a_i, 3] = obj_velo_dir[-1, 1]
            next_state_tensors[a_i, 4] = obj_ab_dir[-1]

        next_state_tensors = math_utils.closest_one_percent(next_state_tensors, 0.01)
        return next_state_tensors

    def get_next_states(self, state3):
        state4_by_actions = torch.zeros((len(self.action_delta), 4, state3.shape[1], state3.shape[2])).to(
            state3.device)

        for a_i in range(len(self.action_delta)):
            next_states = torch.zeros(state3.shape[1], state3.shape[2]).to(state3.device)
            player_pos_delta = torch.tensor(self.action_delta[f'{a_i}']).to(state3.device)
            obj_velocities = reason_utils.get_state_velo(state3)[-1]
            if torch.abs(obj_velocities).max() > 0.3:
                next_states = state3[-1]
            else:
                next_states[0, -2:] = state3[-1, 0, -2:] + player_pos_delta[:2]
                next_states[1, -2:] = state3[-1, 1, -2:] + obj_velocities[1]
                if torch.abs(next_states[:, -2:]).max() > 1:
                    print("")
            state4_by_actions[a_i] = torch.cat((state3, next_states.unsqueeze(0)), dim=0)
        return state4_by_actions

    def explain_for_each_action(self):
        for a_i in range(len(self.action_delta)):
            pass

    def learn_from_dqn(self, state4, dqn_action):
        state_tensor_now = reason_utils.state2analysis_tensor(state4[:-1], 0, 1)
        state_tensor_next = reason_utils.state2analysis_tensor(state4[1:], 0, 1)

        dist_now, explain_now = reason_utils.text_from_tensor(self.o2o_data, state_tensor_now)
        dist_next, explain_next = reason_utils.text_from_tensor(self.o2o_data, state_tensor_next)

        # check if any predicate has been achieved
        closer_beh_mask = (dist_now - dist_next) > 0.01

        if self.previous_pred_mask.sum() > 0:
            self.pwt[self.previous_pred_mask, closer_beh_mask] += 0.01

        previous_pred_mask = dist_now == 0
        if previous_pred_mask.sum() > 0:
            self.previous_pred_mask = previous_pred_mask

    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        action_prob = torch.zeros(1, len(self.args.action_names))
        action_mask = torch.zeros(1, len(self.args.action_names), dtype=torch.bool)
        explains = {"behavior_index": [], "reward": [], 'state': x, "behavior_conf": [], "behavior_action": [],
                    "text": ""}

        if self.last2nd_state is not None:
            state3 = torch.cat((self.last2nd_state, self.last_state, x), dim=0)
        else:
            state3 = torch.cat((x, x, x), dim=0)
        self.last2nd_state = self.last_state
        self.last_state = x

        # use state3 check if satisfy any predicate
        state_tensor = reason_utils.state2analysis_tensor(state3, 0, 1)
        explain_text, onpos_sel_behs, dist_behs, dist_to_o2o_behs, next_possile_beh_explain_text = reason_utils.text_from_tensor(
            self.o2o_data, state_tensor)

        on_pos_best_score = 0
        on_pos_best_beh = None
        on_pos_best_text = ""
        pf_best_score = 0
        pf_best_act = None
        pf_best_text = ""

        # choose the action
        # on_pos = False
        # on_pos_best_text = explain_text
        # if len(onpos_sel_behs) > 0:
        #     on_pos = True
        #     best_beh_index = math_utils.indices_of_maximum(self.o2o_data_weights[onpos_sel_behs]).reshape(-1)
        #     onpos_sel_beh_index = onpos_sel_behs[best_beh_index]

        # on_pos_best_score = self.o2o_data_weights[onpos_sel_beh_index]
        # on_pos_best_beh = onpos_sel_beh_index

        # else using state4 go to next possible predicate

        state4_by_actions = self.get_next_states(state3)
        a_dists = []
        behs = []
        beh_texts = []
        actions = []
        all_dist_behs = []
        action_scores = []
        for a_i in range(len(self.action_delta)):
            state_tensor = reason_utils.state2analysis_tensor(state4_by_actions[a_i], 0, 1)
            next_explain_text, sel_behs, dist_behs, dist_to_o2o_behs2, next_possile_beh_explain_text2 = reason_utils.text_from_tensor(
                self.o2o_data, state_tensor)
            if len(dist_to_o2o_behs2) > 0:
                closest_beh = torch.tensor(dist_to_o2o_behs2).abs().argmin()
                beh_dists = torch.tensor(dist_to_o2o_behs2).abs()
                # assert min(beh_dists) > 0
                dist_inv = 1 - (torch.tensor(beh_dists))
                weigths = dist_inv * self.o2o_data_weights
                # action best behavior

                act_sel_best_beh_index = math_utils.indices_of_maximum(weigths).reshape(-1)[0]

                # act_sel_best_beh_index = 11
                behs.append(act_sel_best_beh_index)
                beh_texts.append(next_possile_beh_explain_text2[act_sel_best_beh_index])
                action_scores.append(weigths[act_sel_best_beh_index].tolist())
            else:
                raise ValueError
                # action_scores.append(0)

        # select closest action

        best_sel_act = math_utils.indices_of_maximum(torch.tensor(action_scores)).reshape(-1, 1)[0]
        # action_prob[0, best_sel_act] = 1
        pf_best_act = best_sel_act
        sel_beh_index = behs[best_sel_act]
        # self.game_o2o_weights[sel_beh_index] += 0.1

        pf_best_text = beh_texts[best_sel_act]
        pf_best_score = action_scores[best_sel_act]
        try:
            if pf_best_score > on_pos_best_score:
                action_prob[0, pf_best_act] = 1
                explains['text'] = pf_best_text
                # self.game_o2o_weights.append(sel_beh_index)
        except TypeError:
            print("")
        if len(onpos_sel_behs) > 0:
            action_prob[0, pf_best_act] = 1
            self.game_o2o_weights += onpos_sel_behs
            explains['text'] = on_pos_best_text

        return action_prob, explains

        action = self.eval_o2o_behavior(state_tensor[-1])
        if action is not None:
            action_prob[0, action.to(torch.int)] = 1
            return action_prob, explains
        else:
            state_time3 = self.get_next_states(state3)
            next_state_tensors = self.get_next_state_tensor(state_time3)

            next_best_dist = 1e+20
            best_action = 0
            for a_i in range(len(self.action_delta)):
                closest_beh, closest_beh_index, closest_beh_dist = self.get_closest_o2o_beh(next_state_tensors[a_i])
                if closest_beh_dist < next_best_dist:
                    best_action = a_i
                    next_best_dist = closest_beh_dist
            action_prob[0, best_action] = 1
            return action_prob, explains
            # find the optimal action

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
                    if beh_conf[pos_index] > 0.0:
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
