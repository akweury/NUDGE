# Created by jing at 27.11.23

import torch

from pi import predicate
from pi.game_settings import get_idx, get_game_info
from pi.utils import smp_utils
from pi.utils.smp_utils import get_all_subsets


def split_data_by_action(states, actions):
    action_types = torch.unique(actions)
    data = {}
    for action_type in action_types:
        index_action = actions == action_type
        states_one_action = states[index_action]
        states_one_action = states_one_action.squeeze()
        if len(states_one_action.shape) == 2:
            states_one_action = states_one_action.unsqueeze(0)
        data[action_type] = states_one_action
    return data


def split_data_by_reward(states, actions, reward):
    reward_types = torch.unique(reward)
    data = {}

    for reward_type in reward_types:
        reward_type_indices = reward == reward_type
        reward_type_states = states[reward_type_indices]
        reward_type_actions = actions[reward_type_indices]
        data[reward_type] = split_data_by_action(reward_type_states, reward_type_actions)
    return data


def gen_mask_name(switch, names):
    mask_name = ''
    for name in names:
        if name in switch:
            mask_name += f'exist_{name}'
        else:
            mask_name += f'not_exist_{name}'
        mask_name += '#'
    mask_name = mask_name[:-1]
    return mask_name


def all_exist_mask(states, obj_names):
    obj_masks = {}
    for obj_i, obj_name in enumerate(obj_names):
        obj_exist = states[:, obj_i, obj_i] > 0
        obj_masks[f'{obj_name}'] = obj_exist

    # states that following different masks
    masks = {}
    switches = get_all_subsets(obj_names)
    for switch in switches:
        switch_masks = []
        for name in obj_names:
            mask = ~obj_masks[name]
            if name in switch:
                mask = ~mask
            switch_masks.append(mask.tolist())
        switch_mask = torch.prod(torch.tensor(switch_masks).float(), dim=0).bool()
        mask_name = gen_mask_name(switch, obj_names)
        masks[mask_name] = switch_mask
        print(f'{mask_name}: {torch.sum(switch_mask)}')

    return masks


def get_all_masks(obj_names):
    masks = []
    obj_combinations = get_all_subsets(obj_names)
    for objs in obj_combinations:
        mask_name = gen_mask_name(objs, obj_names)
        masks.append(mask_name)
    return masks


def get_mask_satisfication(mask_names, states, obj_names):
    obj_masks = {}
    for obj_i, obj_name in enumerate(obj_names):
        obj_exist = states[:, obj_i, obj_i] > 0
        obj_masks[f'{obj_name}'] = obj_exist

    # states that following different masks
    mask_satisfaction_dict = {}
    for mask_name in mask_names:
        res = []
        for obj_name in obj_names:
            m = ~obj_masks[obj_name]
            if obj_name in mask_name:
                m = ~m
            res.append(m.tolist())

        # check if all these states satisfy current mask_name
        satisfication = torch.prod(torch.tensor(res).float(), dim=0).bool()
        mask_satisfaction_dict[mask_name] = satisfication
        print(f'{mask_name}: {torch.sum(satisfication)}')

    return mask_satisfaction_dict


def get_state_mask(state, obj_names):
    mask = torch.zeros(len(obj_names), dtype=torch.bool)
    for obj_i, obj_name in enumerate(obj_names):
        obj_exist = state[obj_i, obj_i] > 0
        mask[obj_i] = obj_exist

    mask_name = ""
    for o_i, obj_name in enumerate(obj_names):
        if mask[o_i]:
            mask_name += f'exist_{obj_name}'
        else:
            mask_name += f'not_exist_{obj_name}'
        mask_name += '#'
    mask_name = mask_name[:-1]
    return mask, mask_name


def check_pred_satisfaction(states, mask, objs, prop_indices):
    all_preds = predicate.get_preds()
    # filter out states following the same mask as current behavior
    obj_A = states[mask, objs[0]]
    obj_B = states[mask, objs[1]]
    sample_num = len(obj_A)
    if sample_num <= 1:
        return [[False] * len(all_preds)] * len(prop_indices), sample_num

    pred_satisfied = []
    for prop_idx in prop_indices:
        data_A = obj_A[:, prop_idx]
        data_B = obj_B[:, prop_idx]
        p_satisfaction = [pred.fit(data_A, data_B, objs) for pred in all_preds]
        pred_satisfied.append(p_satisfaction)
    return pred_satisfied, sample_num


def new_behavior(p_satisfaction, sample_num, objs, prop_idx, action_type, mask_name, reward):
    all_preds = predicate.get_preds()
    is_grounded = False if len(objs) > 1 else True

    behavior = {'preds': all_preds,
                'is_grounded': is_grounded,
                'p_satisfication': p_satisfaction,
                'grounded_types': objs,
                'grounded_prop': prop_idx,
                'action': action_type,
                'mask': mask_name,
                'reward': reward,
                'sample_num': sample_num}

    print(f'new pred, grounded_objs:{objs}, action:{action_type}')
    return behavior


def filter_reward_behaviors(data, obj_names, prop_indices):
    relate_2_obj_types = smp_utils.get_all_2_combinations(obj_names)
    relate_2_prop_types = smp_utils.get_all_subsets(prop_indices, empty_set=False)
    obj_ungrounded_behaviors = []
    for reward, reward_states in data.items():
        if reward == -0.1:
            continue
        for action_type, action_states in reward_states.items():
            masks = all_exist_mask(action_states, obj_names)
            for mask_name, action_mask in masks.items():
                for props in relate_2_prop_types:
                    passed_obj_combs = []
                    sample_nums = []
                    p_satisfactions = []
                    for objs in relate_2_obj_types:
                        p_satisfaction, sample_num = check_pred_satisfaction(action_states, action_mask, objs, props)
                        if (sum([sum(p_satis) for p_satis in p_satisfaction])) > 0:
                            passed_obj_combs.append(objs)
                            sample_nums.append(sample_num)
                            p_satisfactions.append(p_satisfaction)
                    if len(passed_obj_combs) > 0:
                        # obj ungrounded behavior
                        behavior = new_behavior(p_satisfactions, sample_nums, passed_obj_combs, props, action_type,
                                                mask_name, reward)
                        obj_ungrounded_behaviors.append(behavior)
    print(f'ungrounded behaviors: {len(obj_ungrounded_behaviors)}')
    return obj_ungrounded_behaviors


def micro_program2action_behaviors(prop_indices, obj_types, obj_names, data):
    relate_2_obj_types = smp_utils.get_all_2_combinations(obj_types)
    relate_2_prop_types = smp_utils.get_all_2_combinations(prop_indices)
    behaviors = []
    for action, states in data.items():
        masks = all_exist_mask(states, obj_types)
        for obj_types in relate_2_obj_types:
            obj_1_type = obj_types[obj_types[0]]
            obj_2_type = obj_types[obj_types[1]]
            obj_1_indices = obj_names[obj_1_type]
            obj_2_indices = obj_names[obj_2_type]
            for mask_name, mask in masks.items():
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


#
# def micro_program2reward_behaviors(prop_indices, obj_names, data):
#     relate_2_obj_types = smp_utils.get_all_2_combinations(obj_names)
#     relate_2_prop_types = smp_utils.get_all_2_combinations(prop_indices)
#     behaviors = []
#     for reward, action_states in data.items():
#         if reward == -0.1:
#             continue
#         for action, states in action_states.items():
#             masks = all_exist_mask(states, obj_names)
#             for objs in relate_2_obj_types:
#                 for mask_name, mask in masks.items():
#                     for prop_idx in prop_indices:
#                         # select data
#                         data_A = states[mask, objs[0]]
#                         data_B = states[mask, objs[1]]
#
#                         sample_num = len(data_A)
#                         if sample_num <= 1:
#                             continue
#                         data_A = data_A[:, prop_idx]
#                         data_B = data_B[:, prop_idx]
#                         # distinguish predicates
#                         all_preds = predicate.get_preds()
#                         p_satisfication = torch.zeros(len(all_preds), dtype=torch.bool)
#                         for p_i, pred in enumerate(all_preds):
#                             p_satisfication[p_i] = pred.fit(data_A, data_B, objs)
#                         if (p_satisfication.sum()) > 0:
#                             print(f'new pred, grounded_objs:{objs}, action:{action}')
#                             behavior = {'preds': all_preds, 'p_satisfication': p_satisfication,
#                                         'grounded_types': objs, 'grounded_prop': prop_idx,
#                                         'action': action, 'mask': mask_name,
#                                         'reward': reward,
#                                         'sample_num': sample_num}
#                             behaviors.append(behavior)
#
#     return behaviors


def get_all_validations(data):
    indices = list(range(len(data)))
    obj_combinations = get_all_subsets(indices)

    all_combs = torch.zeros(size=(len(obj_combinations), len(data)), dtype=torch.bool)
    for c_i, comb in enumerate(obj_combinations):
        all_combs[c_i, comb] = True

    return all_combs


def gen_all_behaviors(args, obj_names, prop_indices):
    all_smps = []
    relate_2_objs = [[i_1, i_2] for i_1, s_1 in enumerate(obj_names) for i_2, s_2 in
                     enumerate(obj_names) if s_1 != s_2]

    all_preds = predicate.get_preds()
    pred_validations = get_all_validations(all_preds)
    masks = get_all_masks(obj_names)
    for objs in relate_2_objs:
        for mask_name in masks:
            for prop_idx in prop_indices:
                for pred_validation in pred_validations:
                    # distinguish predicates
                    smp = {'preds': all_preds,
                           'grounded_objs': objs,
                           'mask': mask_name,
                           'grounded_prop': prop_idx,
                           'pred_validation': pred_validation
                           }
                    all_smps.append(smp)
    return all_smps


def buffer2behaviors(args, buffer):
    # data preparation
    actions = buffer.actions
    reason_source = buffer.reason_source
    states = buffer.logic_states
    rewards = buffer.rewards
    behaviors = []
    prop_indices = get_idx(args)
    obj_types, obj_names = get_game_info(args)

    data_rewards = split_data_by_reward(states, actions, rewards)
    obj_ungrounded_behaviors = filter_reward_behaviors(data_rewards, obj_types, prop_indices)

    if reason_source is not "random":
        data_actions = split_data_by_action(states, actions)
        behaviors += micro_program2action_behaviors(prop_indices, obj_types, obj_names, data_actions)

    return obj_ungrounded_behaviors, behaviors
