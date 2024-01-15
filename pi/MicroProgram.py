# Created by jing at 04.12.23


import torch
import torch.nn as nn

from pi.utils import smp_utils
from pi import predicate

from src import config


class MicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, args, all_predicates, p_satisfactions, sample_nums, passed_obj_combs, props, action_prob,
                 mask_name, action_mask, reward):
        super().__init__()
        self.args = args
        self.action_prob = action_prob
        self.mask = action_mask
        self.obj_type_existance = smp_utils.mask_name_to_tensor(mask_name, config.mask_splitter)
        self.mask_name = mask_name
        self.obj_type_combs = passed_obj_combs
        self.prop_codes = props
        self.preds = all_predicates
        self.pred_ids = torch.arange(0, len(all_predicates))
        self.sample_nums = sample_nums
        self.p_satisfication = p_satisfactions
        self.obj_type_num = len(passed_obj_combs)
        self.expected_reward = reward

        self.p_spaces = []
        for p_i, pred in enumerate(all_predicates):
            self.p_spaces.append(
                smp_utils.get_param_range(pred.p_bound['min'], pred.p_bound['max'], config.smp_param_unit))

        self.oppm_combs, self.oppm_keys, self.key_nums = smp_utils.arrange_mps(self.obj_type_combs, self.prop_codes,
                                                                               self.obj_type_existance,
                                                                               args.obj_type_indices,
                                                                               self.p_satisfication)

    def check_exists(self, x, obj_type_dict):
        if x.ndim != 3:
            raise ValueError
        state_num = x.size(0)
        mask_batches = self.mask.unsqueeze(0)
        obj_type_exists = torch.ones(size=(state_num, self.obj_type_num), dtype=torch.bool)
        for obj_type_name, obj_type_indices in obj_type_dict.items():
            if obj_type_name in self.obj_type_names:
                obj_indices = [n_i for n_i, name in enumerate(self.obj_type_names) if name == obj_type_name]
                exist_objs = (x[:, obj_type_indices, obj_indices] > 0.8)
                exist_type = exist_objs.prod(dim=-1, keepdims=True).bool()
                obj_type_exists[:, obj_indices] *= exist_type

        mask_batches = torch.repeat_interleave(mask_batches, x.size(0), dim=0)
        exist_res = torch.prod(mask_batches == obj_type_exists, dim=1)
        return exist_res.bool()

    def forward(self, x, obj_type_indices, avg_data=False):
        # game Getout: tensor with size batch_size * 4 * 6
        satisfies = torch.zeros(x.size(0), dtype=torch.bool)
        # if use_given_parameters:
        #     given_parameters = self.p_bound
        # else:
        #     given_parameters = None
        type_1_name = self.obj_type_names[self.type_codes[0]]
        type_1_obj_codes = obj_type_indices[type_1_name]
        type_2_name = self.obj_type_names[self.type_codes[1]]
        type_2_obj_codes = obj_type_indices[type_2_name]

        if len(type_2_obj_codes) > 1 or len(type_1_obj_codes) > 1:
            print("WARNING:")
        # check predicates satisfaction
        for obj_1 in type_1_obj_codes:
            for obj_2 in type_2_obj_codes:
                data_A = x[:, obj_1, self.prop_codes].reshape(-1)
                data_B = x[:, obj_2, self.prop_codes].reshape(-1)

                obj_comb_satisfies = torch.ones(x.size(0), dtype=torch.bool)
                p_spaces = []
                for p_i, pred in enumerate(self.preds):
                    p_space = self.p_spaces[p_i]
                    p_satisfied = self.p_satisfication[p_i]
                    if not p_satisfied:
                        func_satisfy, p_values = torch.ones(data_A.size()).bool(), torch.zeros(size=data_A.size())
                    else:
                        func_satisfy, p_values = pred.eval(data_A, data_B, p_space)
                    p_spaces.append(p_values.unsqueeze(0))

                    # satisfy all
                    obj_comb_satisfies *= func_satisfy

                # satisfy any
                satisfies += obj_comb_satisfies

        # check mask satisfaction
        exist_satisfy = self.check_exists(x, obj_type_indices)

        # satisfy all
        satisfies *= exist_satisfy

        # return action probs
        action_probs = torch.zeros(x.size(0), len(self.action))
        action_probs[satisfies] += self.action
        action_probs[satisfies] = action_probs[satisfies] / (action_probs[satisfies] + 1e-20)

        return action_probs, p_spaces


class UngroundedMicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, args, all_predicates, p_satisfactions, sample_nums, passed_obj_combs, props, action_type,
                 mask_name, action_mask, reward):
        super().__init__()
        self.args = args
        self.action = action_type
        self.mask = action_mask
        self.obj_type_existance = smp_utils.mask_name_to_tensor(mask_name, config.mask_splitter)
        self.mask_name = mask_name
        self.obj_type_combs = passed_obj_combs
        self.prop_codes = props
        self.preds = all_predicates
        self.pred_ids = torch.arange(0, len(all_predicates))
        self.sample_nums = sample_nums

        self.p_satisfication = p_satisfactions
        self.obj_type_num = len(passed_obj_combs)
        self.expected_reward = reward

        self.p_spaces = []
        for p_i, pred in enumerate(all_predicates):
            self.p_spaces.append(
                smp_utils.get_param_range(pred.p_bound['min'], pred.p_bound['max'], config.smp_param_unit))

        self.oppm_combs, self.oppm_keys, self.key_nums = smp_utils.arrange_mps(self.obj_type_combs, self.prop_codes,
                                                                               self.obj_type_existance,
                                                                               args.obj_type_indices,
                                                                               self.p_satisfication)
        self.id = str(reward) + str(action_type) + str(mask_name) + str(props)

    # def check_exists(self, x, obj_type_dict):
    #     if x.ndim != 3:
    #         raise ValueError
    #     state_num = x.size(0)
    #     mask_batches = self.mask.unsqueeze(0)
    #     obj_type_exists = torch.ones(size=(state_num, self.obj_type_num), dtype=torch.bool)
    #     for obj_type_name, obj_type_indices in obj_type_dict.items():
    #         if obj_type_name in self.args.obj_type_names:
    #             obj_indices = [n_i for n_i, name in enumerate(self.obj_type_names) if name == obj_type_name]
    #             exist_objs = (x[:, obj_type_indices, obj_indices] > 0.8)
    #             exist_type = exist_objs.prod(dim=-1, keepdims=True).bool()
    #             obj_type_exists[:, obj_indices] *= exist_type
    #
    #     mask_batches = torch.repeat_interleave(mask_batches, x.size(0), dim=0)
    #     exist_res = torch.prod(mask_batches == obj_type_exists, dim=1)
    #     return exist_res.bool()

    def forward(self, x):
        # given game a state, predict an action
        # game Getout: tensor with size batch_size * 4 * 6

        satisfactions = []
        for oppm_comb in self.oppm_combs:
            satisfaction = smp_utils.oppm_eval(x, oppm_comb, self.oppm_keys, self.preds, self.p_spaces,
                                               self.args.obj_type_indices, self.args.obj_type_names)
            satisfactions.append(satisfaction)

        return satisfactions


class Behavior():
    """ generate one micro-program
    """

    def __init__(self, fact, action, reward):
        super().__init__()
        self.fact = fact
        self.action = action
        self.reward = reward

    def update_pred_params(self, preds, x):
        satisfactions = torch.ones(len(self.fact["preds"]), len(x), dtype=torch.bool)
        params = torch.zeros(len(self.fact["preds"]), len(x))
        for i in range(len(preds)):
            if self.fact["pred_tensors"][i]:
                obj_0 = self.fact["objs"][0]
                obj_1 = self.fact["objs"][1]
                prop = self.fact["props"]
                data_A = x[:, obj_0, prop].reshape(-1)
                data_B = x[:, obj_1, prop].reshape(-1)

                preds[i].update_space(data_A, data_B)
                # func_satisfy, p_values = pred.eval(data_A, data_B, p_space)

        return satisfactions, params


class SymbolicMicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.action_num = len(args.action_names)
        self.ungrounded_behaviors = []
        self.buffer = None
        self.data_rewards = None
        self.data_actions = None
        self.data_combs = None
        self.obj_ungrounded_behavior_ids = []

    def load_buffer(self, buffer):
        print(f'- SMP new buffer, total states: {len(buffer.logic_states)}')
        self.buffer = buffer
        self.data_rewards = smp_utils.split_data_by_reward(self.buffer.logic_states, self.buffer.actions,
                                                           self.buffer.rewards, self.action_num)
        self.data_actions = smp_utils.split_data_by_action(self.buffer.logic_states, self.buffer.actions,
                                                           self.action_num)
        if len(self.buffer.rewards) > 0:
            self.data_combs = smp_utils.comb_buffers(self.buffer.logic_states, self.buffer.actions, self.buffer.rewards)

    def teacher_searching(self, obj_types):
        # strategy: action, objects, mask, properties, if predicates are valid
        behaviors = []
        for action, action_states in self.data_actions.items():
            all_masks = smp_utils.all_exist_mask(action_states, obj_types)
            for fact in self.facts:
                satisfy = smp_utils.satisfy_fact(fact, action_states, all_masks)
                if satisfy:
                    behavior = Behavior(fact, action, None)
                    smp_utils.update_pred_parameters(self.preds, action_states, behavior)
                    behaviors.append(behavior)
        return behaviors

    def forward_searching(self, relate_2_obj_types, relate_2_prop_types, obj_types):
        # ungrounded behaviors:
        # Existing at least one predicate is true,
        # but not know which part of the data is the reason.

        obj_grounded_behaviors = []
        for reward, reward_states in self.data_rewards.items():
            if reward == -0.1:
                continue
            for action_prob, action_states in reward_states.items():
                masks = smp_utils.all_exist_mask(action_states, obj_types)
                for mask_name, action_mask in masks.items():
                    if action_mask.sum() == 0:
                        continue
                    for props in relate_2_prop_types:
                        passed_obj_combs = []
                        sample_nums = []
                        p_satisfactions = []
                        all_predicates = predicate.get_preds(len(props))
                        for objs in relate_2_obj_types:
                            p_satisfaction, sample_num = smp_utils.check_pred_satisfaction(action_states,
                                                                                           all_predicates,
                                                                                           action_mask, objs, props)
                            if (sum(p_satisfaction)) > 0:
                                passed_obj_combs.append(objs)
                                sample_nums.append(sample_num)
                                p_satisfactions.append(p_satisfaction)
                        if len(passed_obj_combs) > 1:
                            # obj ungrounded behavior
                            behavior = UngroundedMicroProgram(self.args, all_predicates, p_satisfactions, sample_nums,
                                                              passed_obj_combs, props, action_prob, mask_name,
                                                              action_mask, reward)
                            if behavior.id not in self.obj_ungrounded_behavior_ids:
                                self.ungrounded_behaviors.append(behavior)
                                self.obj_ungrounded_behavior_ids.append(behavior.id)
                            else:
                                print("should not be called.")
                        elif len(passed_obj_combs) == 1:
                            behavior = MicroProgram(self.args, all_predicates, p_satisfactions, sample_nums,
                                                    passed_obj_combs, props, action_prob, mask_name, action_mask,
                                                    reward)
                            obj_grounded_behaviors.append(behavior)
                print(f"reward: {reward}, action: {action_prob}, states: {len(action_states)}")
        print(f'forward searched ungrounded behaviors: {len(self.ungrounded_behaviors)}')
        print(f'forward searched grounded behaviors: {len(obj_grounded_behaviors)}')
        return obj_grounded_behaviors

    def backward_searching(self):
        print(f"- backward searching ...")
        for reward, reward_states in self.data_rewards.items():
            if reward == -0.1:
                continue
            for action_prob, action_states in reward_states.items():
                for behavior in self.ungrounded_behaviors:

                    if behavior.expected_reward == reward and torch.equal(behavior.action, action_prob):
                        # check which explains can satisfy more states

                        satisfactions = behavior(action_states)
                        # filter out wrong groundings, until only one option left
                        # update behavior's oppm combs
                        behavior.oppm_combs = self.grounding(behavior.oppm_combs, satisfactions)
                        # top scored explains shall be kept
                        # how to remove them?
        new_ungrounded_behaviors = []
        for behavior in self.ungrounded_behaviors:
            if len(behavior.oppm_combs) > 0:
                new_ungrounded_behaviors.append(behavior)
        self.ungrounded_behaviors = new_ungrounded_behaviors

    def backward_searching2(self):
        print(f"- backward searching 2...")
        # consider batching evaluation
        for behavior in self.ungrounded_behaviors:
            satisfactions_behavior = []
            satisfactions_mask = []
            for data in self.data_combs:
                state, action, reward = data
                if behavior.expected_reward == reward:

                    satisfactions_data = []
                    same_action_id = torch.argmax(behavior.action).item() == action
                    same_mask = smp_utils.mask_name_from_state(state, self.args.obj_type_names,
                                                               config.mask_splitter) == behavior.mask_name
                    if same_action_id and same_mask:
                        satisfactions_mask.append(True)
                        for oppm_comb in behavior.oppm_combs:
                            comb_obj_id = [oppm_comb[idx] for idx in behavior.oppm_keys["obj"]]
                            comb_prop_id = [oppm_comb[idx] for idx in behavior.oppm_keys["prop"]]
                            comb_preds = [oppm_comb[idx] for idx in behavior.oppm_keys["pred"]]
                            state_preds, _ = smp_utils.check_pred_satisfaction(state, behavior.preds, None, comb_obj_id,
                                                                               comb_prop_id, behavior.p_spaces,
                                                                               mode="eval")
                            same_pred = state_preds == comb_preds
                            if same_pred:
                                satisfactions_data.append(True)
                            else:
                                satisfactions_data.append(False)
                    else:
                        satisfactions_mask.append(False)
                else:
                    satisfactions_data = [False] * len(behavior.oppm_combs)
                    satisfactions_mask.append(False)

                satisfactions_behavior.append(satisfactions_data)
            passed_states = [satisfactions_behavior[s_i] for s_i, state in enumerate(satisfactions_mask) if state]

            behavior.oppm_combs = self.grounding(behavior.oppm_combs, passed_states)

        new_ungrounded_behaviors = []
        for behavior in self.ungrounded_behaviors:
            if len(behavior.oppm_combs) > 0:
                new_ungrounded_behaviors.append(behavior)
        self.ungrounded_behaviors = new_ungrounded_behaviors

    def grounding(self, combs, satisfactions):

        satisfaction_count = torch.tensor(satisfactions).float()
        if len(satisfaction_count.size()) == 2:
            satisfaction_count = satisfaction_count.sum(0)
        if len(combs) != len(satisfaction_count):
            print("Warning:")
        satisfactions = satisfaction_count > 0
        grounded_combs = [combs[s_i] for s_i, satisfaction in enumerate(satisfactions) if satisfaction]
        print(f"behavior grounded reasons from {len(combs)} to {len(grounded_combs)}")
        return grounded_combs

    def programming(self, obj_types, prop_indices):
        relate_2_obj_types = smp_utils.get_all_2_combinations(obj_types)
        relate_2_prop_types = smp_utils.get_all_subsets(prop_indices, empty_set=False)
        self.preds = predicate.get_preds()
        self.facts = smp_utils.get_smp_facts(obj_types, relate_2_obj_types, relate_2_prop_types, self.preds)

        behaviors = self.teacher_searching(obj_types)
        # obj_grounded_behaviors = self.forward_searching(relate_2_obj_types, relate_2_prop_types, obj_types)
        # self.backward_searching()
        # self.backward_searching2()

        return behaviors

    # def behaviors_from_actions(self, relate_2_obj_types, relate_2_prop_types, obj_types):
    #     behaviors = []
    #     for action, states in self.data_actions.items():
    #         masks = smp_utils.all_exist_mask(states, obj_types)
    #         for mask_name, mask in masks.items():
    #             for obj_types in relate_2_obj_types:
    #                 obj_1_type = obj_types[obj_types[0]]
    #                 obj_2_type = obj_types[obj_types[1]]
    #                 obj_1_indices = obj_types[obj_1_type]
    #                 obj_2_indices = obj_types[obj_2_type]
    #                 for prop_types in relate_2_prop_types:
    #                     # as long as any two objs satisfied
    #                     all_preds = predicate.get_preds()
    #                     p_satisfication = torch.zeros(len(all_preds), dtype=torch.bool)
    #                     for obj_1_index in obj_1_indices:
    #                         for obj_2_index in obj_2_indices:
    #                             # select data
    #                             if obj_2_index >= 4:
    #                                 raise ValueError
    #                             data_A = states[mask, obj_1_index]
    #                             data_B = states[mask, obj_2_index]
    #                             if len(data_A) == 0:
    #                                 continue
    #                             data_A = data_A[:, prop_types]
    #                             data_B = data_B[:, prop_types]
    #                             # distinguish predicates
    #                             for p_i, pred in enumerate(all_preds):
    #                                 p_satisfication[p_i] += pred.fit(data_A, data_B, obj_types)
    #
    #                     if (p_satisfication.sum()) > 0:
    #                         print(f'new pred, grounded_objs:{obj_types}, action:{action}')
    #                         behavior = {'preds': all_preds,
    #                                     'p_satisfication': p_satisfication,
    #                                     'is_grounded': True,
    #                                     'grounded_types': obj_types,
    #                                     'grounded_prop': prop_types,
    #                                     'action': action,
    #                                     'mask': mask_name}
    #                         behaviors.append(behavior)
    #
    #     return behaviors

    # def check_exists(self, x, obj_type_dict):
    #     if x.ndim != 3:
    #         raise ValueError
    #     state_num = x.size(0)
    #     mask_batches = self.mask.unsqueeze(0)
    #     obj_type_exists = torch.ones(size=(state_num, self.obj_type_num), dtype=torch.bool)
    #     for obj_type_name, obj_type_indices in obj_type_dict.items():
    #         if obj_type_name in self.obj_type_names:
    #             obj_indices = [n_i for n_i, name in enumerate(self.obj_type_names) if name == obj_type_name]
    #             exist_objs = (x[:, obj_type_indices, obj_indices] > 0.8)
    #             exist_type = exist_objs.prod(dim=-1, keepdims=True).bool()
    #             obj_type_exists[:, obj_indices] *= exist_type
    #
    #     mask_batches = torch.repeat_interleave(mask_batches, x.size(0), dim=0)
    #     exist_res = torch.prod(mask_batches == obj_type_exists, dim=1)
    #     return exist_res.bool()

    # def forward(self, x, obj_type_indices, avg_data=False):
    #     # game Getout: tensor with size batch_size * 4 * 6
    #     action_probs = torch.zeros(len(self.type_codes), x.size(0), len(self.action))
    #     p_spaces = []
    #     for t_i, type_code in enumerate(self.type_codes):
    #         t_i_p_spaces = []
    #         satisfies = torch.zeros(x.size(0), dtype=torch.bool)
    #         type_1_obj_codes = obj_type_indices[self.obj_type_names[type_code[0]]]
    #         type_2_obj_codes = obj_type_indices[self.obj_type_names[type_code[1]]]
    #         # check predicates satisfaction
    #         for obj_1 in type_1_obj_codes:
    #             for obj_2 in type_2_obj_codes:
    #                 for prop_code in self.prop_codes:
    #                     data_A = x[:, obj_1, prop_code].reshape(-1)
    #                     data_B = x[:, obj_2, prop_code].reshape(-1)
    #
    #                     obj_comb_satisfies = torch.ones(x.size(0), dtype=torch.bool)
    #
    #                     for p_i, pred in enumerate(self.preds):
    #                         p_space = self.p_spaces[t_i][p_i]
    #                         p_satisfied = self.p_satisfication[p_i]
    #                         if not p_satisfied:
    #                             func_satisfy, p_values = torch.ones(data_A.size()).bool(), torch.zeros(
    #                                 size=data_A.size())
    #                         else:
    #                             func_satisfy, p_values = pred.eval(data_A, data_B, p_space)
    #                         t_i_p_spaces.append(p_values.unsqueeze(0))
    #
    #                         # satisfy all
    #                         obj_comb_satisfies *= func_satisfy
    #
    #                     # satisfy any
    #                     satisfies += obj_comb_satisfies
    #
    #         # check mask satisfaction
    #         exist_satisfy = self.check_exists(x, obj_type_indices)
    #
    #         # satisfy all
    #         satisfies *= exist_satisfy
    #
    #         # return action probs
    #         action_probs[t_i, satisfies] += self.action
    #         action_probs[t_i, satisfies] = action_probs[t_i, satisfies] / (action_probs[t_i, satisfies] + 1e-20)
    #         p_spaces.append(t_i_p_spaces)
    #
    #     return action_probs, p_spaces
