# Created by jing at 23.12.23
import torch


def get_param_range(min, max, unit):
    length = (max - min) // unit
    if length == 0:
        return torch.zeros(1)

    space = torch.zeros(int(length))
    for v_i in range(len(space)):
        space[v_i] = min + unit * v_i
    return space


def get_all_2_combinations(data):
    all_combinations = [[i_1, i_2] for i_1, s_1 in enumerate(data) for i_2, s_2 in
                        enumerate(data) if s_1 != s_2]
    return all_combinations


def get_all_subsets(input_list, empty_set=True):
    result = [[]]

    for elem in input_list:
        new_subsets = [subset + [elem] for subset in result]
        result.extend(new_subsets)

    if not empty_set:
        result.remove([])
    return result


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


def check_pred_satisfaction(states, all_preds, mask, objs, prop_indices):
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
