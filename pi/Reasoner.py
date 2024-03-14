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

    def update(self, args, behaviors, requirement):
        self.requirement = requirement
        self.aligning = False
        self.unaligned = False
        self.kill_enemy = False
        self.target_obj = 1
        self.next_target = self.requirement[0]
        self.unaligned_target = None
        self.sub_align_axis = None
        self.align_to_sub_object = False
        self.unaligned_align_to_sub_object = False
        # self.o2o_data = o2o_data[1:]
        # self.o2o_achieved = torch.ones(len(self.o2o_data), dtype=torch.bool)
        # self.o2o_data_weights = torch.ones(len(o2o_data))
        # self.game_o2o_weights = []
        # self.previous_pred_mask = torch.zeros(len(self.o2o_data), dtype=torch.bool)
        # self.pwt = torch.zeros(len(self.o2o_data), len(self.o2o_data))

        if args is not None:
            self.args = args
        if behaviors is not None and len(behaviors) > 0:
            self.behaviors = behaviors
            self.set_model()

    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        action_prob = torch.zeros(1, len(self.args.action_names))
        action_mask = torch.zeros(1, len(self.args.action_names), dtype=torch.bool)
        explains = {"behavior_index": [], "reward": [], 'state': x, "behavior_conf": [], "behavior_action": [],
                    "text": ""}

        if self.kill_enemy:
            target_obj = self.unaligned_target
            action = self.get_kill_action(x[0, 0, -2:], x[0, target_obj, -2:])
        elif self.unaligned_target is not None:
            target_obj = self.unaligned_target
            avoid_axis = self.unaligned_axis
            action = self.get_avoidance_action(x[0, 0], avoid_axis, x[0, target_obj, avoid_axis])

        else:
            # determine align object
            target_obj = self.next_target
            align_axis = self.align_axis
            action = self.get_closest_action(x[0, 0], align_axis, x[0, target_obj, align_axis])
        action_prob[:, action] = 1
        return action_prob, explains

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

    def get_next_states_boxing(self, x):
        batch_size = len(self.args.action_names)
        batch_current_state = torch.repeat_interleave(x, batch_size, 0).reshape(batch_size, -1)
        batch_actions = torch.arange(batch_size).unsqueeze(1)

        batch_next_states = self.state_estimator(batch_current_state, batch_actions).reshape(batch_size, x.shape[1],
                                                                                             x.shape[2])
        batch_current_states = torch.repeat_interleave(x.unsqueeze(0), batch_size, 0)
        batch_last_states = torch.repeat_interleave(self.last_state.unsqueeze(0), batch_size, 0)
        batch_last2nd_states = torch.repeat_interleave(self.last2nd_state.unsqueeze(0), batch_size, 0)
        batch_state4 = torch.cat(
            (batch_last2nd_states, batch_last_states, batch_current_states, batch_next_states.unsqueeze(1)), dim=1)
        return batch_state4

    def get_next_states_kangaroo(self, x):
        batch_size = len(self.args.action_names)
        batch_current_state = torch.repeat_interleave(x, batch_size, 0).reshape(batch_size, -1)
        batch_actions = torch.arange(batch_size).unsqueeze(1).to(x.device)

        batch_next_states = self.state_estimator(batch_current_state, batch_actions).reshape(batch_size, x.shape[1],
                                                                                             x.shape[2])

        return batch_next_states

    def get_next_states_pong(self, x):
        # state4_by_actions = torch.zeros((len(self.args.action_names), 4, state3.shape[1], state3.shape[2])).to(
        #     state3.device)
        #
        # for a_i in range(len(self.action_delta)):
        #     next_states = torch.zeros(state3.shape[1], state3.shape[2]).to(state3.device)
        #     player_pos_delta = torch.tensor(self.action_delta[f'{a_i}']).to(state3.device)
        #     obj_velocities = reason_utils.get_state_velo(state3)[-1]
        #     if torch.abs(obj_velocities).max() > 0.3:
        #         next_states = state3[-1]
        #     else:
        #         next_states[0, -2:] = state3[-1, 0, -2:] + player_pos_delta[:2]
        #         next_states[1, -2:] = state3[-1, 1, -2:] + obj_velocities[1]
        #         if torch.abs(next_states[:, -2:]).max() > 1:
        #             print("")
        #     state4_by_actions[a_i] = torch.cat((state3, next_states.unsqueeze(0)), dim=0)
        batch_size = len(self.args.action_names)
        batch_current_state = torch.repeat_interleave(x, batch_size, 0).reshape(batch_size, -1)
        batch_actions = torch.arange(batch_size).unsqueeze(1)

        batch_next_states = self.state_estimator(batch_current_state, batch_actions).reshape(batch_size, x.shape[1],
                                                                                             x.shape[2])
        batch_current_states = torch.repeat_interleave(x.unsqueeze(0), batch_size, 0)
        batch_last_states = torch.repeat_interleave(self.last_state.unsqueeze(0), batch_size, 0)
        batch_last2nd_states = torch.repeat_interleave(self.last2nd_state.unsqueeze(0), batch_size, 0)
        batch_state4 = torch.cat(
            (batch_last2nd_states, batch_last_states, batch_current_states, batch_next_states.unsqueeze(1)), dim=1)
        return batch_state4

    def explain_for_each_action(self):
        for a_i in range(len(self.action_delta)):
            pass

    def learn_from_dqn(self, state4, dqn_action):
        if self.args.m == "Boxing":
            state_tensor_now = reason_utils.state2analysis_tensor_boxing(state4[:-1], 0, 1)
            state_tensor_next = reason_utils.state2analysis_tensor_boxing(state4[1:], 0, 1)
        elif self.args.m == "Pong":
            state_tensor_now = reason_utils.state2analysis_tensor_pong(state4[:-1], 0, 1)
            state_tensor_next = reason_utils.state2analysis_tensor_pong(state4[1:], 0, 1)
        else:
            raise ValueError

        dist_now, explain_now = reason_utils.text_from_tensor(self.o2o_data, state_tensor_now, self.args.prop_explain)
        dist_next, explain_next = reason_utils.text_from_tensor(self.o2o_data, state_tensor_next,
                                                                self.args.prop_explain)

        # check if any predicate has been achieved
        closer_beh_mask = (dist_now - dist_next) > 0.01

        if self.previous_pred_mask.sum() > 0:
            repeat_prevpredmask = torch.repeat_interleave(self.previous_pred_mask.reshape(-1, 1),
                                                          len(self.previous_pred_mask), 1)
            repeat_closer_beh_mask = torch.repeat_interleave(closer_beh_mask.reshape(1, -1), len(closer_beh_mask), 0)
            self.pwt[repeat_prevpredmask * repeat_closer_beh_mask] += 0.01

        previous_pred_mask = dist_now == 0
        if previous_pred_mask.sum() > 0:
            self.previous_pred_mask = previous_pred_mask

    def get_closest_action(self, x, align_axis, target_pos):
        # if self.args.m == "Kangaroo":
        #     next_states = self.get_next_states_kangaroo(x)
        # else:
        #     raise ValueError
        dist_actions = torch.zeros(len(self.args.action_names))
        for a_i in range(len(self.args.action_names)):
            player_pos = x[align_axis].clone()
            if align_axis == -2:
                if "left" in self.args.action_names[a_i]:
                    player_pos -= 0.01
                if "right" in self.args.action_names[a_i]:
                    player_pos += 0.01
            elif align_axis == -1:
                if "up" in self.args.action_names[a_i]:
                    player_pos -= 0.01
                if "down" in self.args.action_names[a_i]:
                    player_pos += 0.00
            else:
                raise ValueError
            dist = torch.abs(target_pos - player_pos).sum()
            dist_actions[a_i] = dist
        best_action = torch.argmin(dist_actions)
        return best_action

    def get_avoidance_action(self, x, avoid_axis, target_pos):
        # if self.args.m == "Kangaroo":
        #     next_states = self.get_next_states_kangaroo(x)
        # else:
        #     raise ValueError
        dist_actions = torch.zeros(len(self.args.action_names))
        for a_i in range(len(self.args.action_names)):
            player_pos = x[avoid_axis].clone()
            if avoid_axis == -2:
                if "left" in self.args.action_names[a_i]:
                    player_pos -= 0.01
                if "right" in self.args.action_names[a_i]:
                    player_pos += 0.01
            elif avoid_axis == -1:
                if "up" in self.args.action_names[a_i]:
                    player_pos -= 0.01
                if "down" in self.args.action_names[a_i]:
                    player_pos += 0.00
            else:
                raise ValueError
            dist = torch.abs(target_pos - player_pos).sum()
            dist_actions[a_i] = dist
        best_action = torch.argmax(dist_actions)
        return best_action

    def get_kill_action(self, player_pos, target_pos):
        try:
            pos_diff = target_pos - player_pos
        except RuntimeError:
            print("")
        if pos_diff[0] > 0:
            dir_x = "right"
        else:
            dir_x = "left"

        if pos_diff[1] > 0:
            dir_y = "down"
        else:
            dir_y = "up"

        best_action = 1
        for a_i in range(len(self.args.action_names)):
            if dir_x in self.args.action_names[a_i] and dir_y in self.args.action_names[a_i]:
                best_action = a_i
        return best_action
