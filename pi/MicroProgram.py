# Created by jing at 04.12.23


import torch
import torch.nn as nn
from pi.utils import smp_utils
from pi import predicate
from src import config


class MicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, all_preds, obj_types, behavior):
        super().__init__()
        self.obj_type_names = obj_type_name
        self.action = action
        self.mask = mask
        self.type_codes = type_codes
        self.prop_codes = prop_codes
        self.preds = all_preds
        self.p_spaces = p_spaces
        self.p_satisfication = behavior["p_satisfication"]
        self.obj_type_num = len(obj_type_name)
        self.reward = behavior["reward"]

        assert len(self.prop_codes) == 1

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

    def __init__(self, all_predicates, p_satisfactions, sample_nums, passed_obj_combs, props, action_type, mask_name,
                 action_mask, reward):
        super().__init__()

        self.action = action_type
        self.mask = action_mask
        self.mask_name = mask_name
        self.type_codes = passed_obj_combs
        self.prop_codes = props
        self.preds = all_predicates
        self.sample_nums = sample_nums

        self.p_satisfication = p_satisfactions
        self.obj_type_num = len(passed_obj_combs)
        self.expected_reward = reward

        p_spaces = []
        for p_i, pred in enumerate(all_predicates):
            p_spaces.append(smp_utils.get_param_range(pred.p_bound['min'], pred.p_bound['max'], config.smp_param_unit))
        self.p_spaces = [p_spaces] * len(passed_obj_combs)

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
        action_probs = torch.zeros(len(self.type_codes), x.size(0), len(self.action))
        p_spaces = []
        for t_i, type_code in enumerate(self.type_codes):
            t_i_p_spaces = []
            satisfies = torch.zeros(x.size(0), dtype=torch.bool)
            type_1_obj_codes = obj_type_indices[self.obj_type_names[type_code[0]]]
            type_2_obj_codes = obj_type_indices[self.obj_type_names[type_code[1]]]
            # check predicates satisfaction
            for obj_1 in type_1_obj_codes:
                for obj_2 in type_2_obj_codes:
                    for prop_code in self.prop_codes:
                        data_A = x[:, obj_1, prop_code].reshape(-1)
                        data_B = x[:, obj_2, prop_code].reshape(-1)

                        obj_comb_satisfies = torch.ones(x.size(0), dtype=torch.bool)

                        for p_i, pred in enumerate(self.preds):
                            p_space = self.p_spaces[t_i][p_i]
                            p_satisfied = self.p_satisfication[p_i]
                            if not p_satisfied:
                                func_satisfy, p_values = torch.ones(data_A.size()).bool(), torch.zeros(
                                    size=data_A.size())
                            else:
                                func_satisfy, p_values = pred.eval(data_A, data_B, p_space)
                            t_i_p_spaces.append(p_values.unsqueeze(0))

                            # satisfy all
                            obj_comb_satisfies *= func_satisfy

                        # satisfy any
                        satisfies += obj_comb_satisfies

            # check mask satisfaction
            exist_satisfy = self.check_exists(x, obj_type_indices)

            # satisfy all
            satisfies *= exist_satisfy

            # return action probs
            action_probs[t_i, satisfies] += self.action
            action_probs[t_i, satisfies] = action_probs[t_i, satisfies] / (action_probs[t_i, satisfies] + 1e-20)
            p_spaces.append(t_i_p_spaces)

        return action_probs, p_spaces


class SymbolicMicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, args, buffer):
        super().__init__()
        self.args = args
        self.buffer = buffer
        self.data_rewards = smp_utils.split_data_by_reward(self.buffer.logic_states, self.buffer.actions,
                                                           self.buffer.rewards)
        self.data_actions = smp_utils.split_data_by_action(self.buffer.logic_states, self.buffer.actions)

    def behaviors_from_rewards(self, relate_2_obj_types, relate_2_prop_types, obj_types):
        obj_ungrounded_behaviors = []
        for reward, reward_states in self.data_rewards.items():
            if reward == -0.1:
                continue
            for action_type, action_states in reward_states.items():
                masks = smp_utils.all_exist_mask(action_states, obj_types)
                for mask_name, action_mask in masks.items():
                    for props in relate_2_prop_types:
                        passed_obj_combs = []
                        sample_nums = []
                        p_satisfactions = []
                        all_predicates = predicate.get_preds()
                        for objs in relate_2_obj_types:
                            p_satisfaction, sample_num = smp_utils.check_pred_satisfaction(action_states,
                                                                                           all_predicates,
                                                                                           action_mask, objs, props)
                            if (sum([sum(p_satis) for p_satis in p_satisfaction])) > 0:
                                passed_obj_combs.append(objs)
                                sample_nums.append(sample_num)
                                p_satisfactions.append(p_satisfaction)
                        if len(passed_obj_combs) > 0:
                            # obj ungrounded behavior
                            behavior = UngroundedMicroProgram(all_predicates, p_satisfactions, sample_nums,
                                                              passed_obj_combs, props, action_type, mask_name,
                                                              action_mask, reward)
                            obj_ungrounded_behaviors.append(behavior)
        print(f'ungrounded behaviors: {len(obj_ungrounded_behaviors)}')
        return obj_ungrounded_behaviors

    def behaviors_from_actions(self, relate_2_obj_types, relate_2_prop_types, obj_types):
        behaviors = []
        for action, states in self.data_actions.items():
            masks = smp_utils.all_exist_mask(states, obj_types)
            for mask_name, mask in masks.items():
                for obj_types in relate_2_obj_types:
                    obj_1_type = obj_types[obj_types[0]]
                    obj_2_type = obj_types[obj_types[1]]
                    obj_1_indices = obj_types[obj_1_type]
                    obj_2_indices = obj_types[obj_2_type]
                    for prop_types in relate_2_prop_types:
                        # as long as any two objs satisfied
                        all_preds = predicate.get_preds()
                        p_satisfication = torch.zeros(len(all_preds), dtype=torch.bool)
                        for obj_1_index in obj_1_indices:
                            for obj_2_index in obj_2_indices:
                                # select data
                                if obj_2_index >= 4:
                                    raise ValueError
                                data_A = states[mask, obj_1_index]
                                data_B = states[mask, obj_2_index]
                                if len(data_A) == 0:
                                    continue
                                data_A = data_A[:, prop_types]
                                data_B = data_B[:, prop_types]
                                # distinguish predicates
                                for p_i, pred in enumerate(all_preds):
                                    p_satisfication[p_i] += pred.fit(data_A, data_B, obj_types)

                        if (p_satisfication.sum()) > 0:
                            print(f'new pred, grounded_objs:{obj_types}, action:{action}')
                            behavior = {'preds': all_preds,
                                        'p_satisfication': p_satisfication,
                                        'is_grounded': True,
                                        'grounded_types': obj_types,
                                        'grounded_prop': prop_types,
                                        'action': action,
                                        'mask': mask_name}
                            behaviors.append(behavior)

        return behaviors

    def programming(self, obj_types, prop_indices):
        relate_2_obj_types = smp_utils.get_all_2_combinations(obj_types)
        relate_2_prop_types = smp_utils.get_all_subsets(prop_indices, empty_set=False)

        obj_ungrounded_behaviors = self.behaviors_from_rewards(relate_2_obj_types, relate_2_prop_types, obj_types)
        grounded_behaviors = self.behaviors_from_actions(relate_2_obj_types, relate_2_prop_types, obj_types)

        return grounded_behaviors, obj_ungrounded_behaviors

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
        action_probs = torch.zeros(len(self.type_codes), x.size(0), len(self.action))
        p_spaces = []
        for t_i, type_code in enumerate(self.type_codes):
            t_i_p_spaces = []
            satisfies = torch.zeros(x.size(0), dtype=torch.bool)
            type_1_obj_codes = obj_type_indices[self.obj_type_names[type_code[0]]]
            type_2_obj_codes = obj_type_indices[self.obj_type_names[type_code[1]]]
            # check predicates satisfaction
            for obj_1 in type_1_obj_codes:
                for obj_2 in type_2_obj_codes:
                    for prop_code in self.prop_codes:
                        data_A = x[:, obj_1, prop_code].reshape(-1)
                        data_B = x[:, obj_2, prop_code].reshape(-1)

                        obj_comb_satisfies = torch.ones(x.size(0), dtype=torch.bool)

                        for p_i, pred in enumerate(self.preds):
                            p_space = self.p_spaces[t_i][p_i]
                            p_satisfied = self.p_satisfication[p_i]
                            if not p_satisfied:
                                func_satisfy, p_values = torch.ones(data_A.size()).bool(), torch.zeros(
                                    size=data_A.size())
                            else:
                                func_satisfy, p_values = pred.eval(data_A, data_B, p_space)
                            t_i_p_spaces.append(p_values.unsqueeze(0))

                            # satisfy all
                            obj_comb_satisfies *= func_satisfy

                        # satisfy any
                        satisfies += obj_comb_satisfies

            # check mask satisfaction
            exist_satisfy = self.check_exists(x, obj_type_indices)

            # satisfy all
            satisfies *= exist_satisfy

            # return action probs
            action_probs[t_i, satisfies] += self.action
            action_probs[t_i, satisfies] = action_probs[t_i, satisfies] / (action_probs[t_i, satisfies] + 1e-20)
            p_spaces.append(t_i_p_spaces)

        return action_probs, p_spaces
