# Created by jing at 23.12.23
import torch

from src import config

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


def split_data_by_action(states, actions, action_num):
    action_types = torch.unique(actions)
    data = {}
    for action_type in action_types:
        index_action = actions == action_type
        states_one_action = states[index_action]
        states_one_action = states_one_action.squeeze()
        if len(states_one_action.shape) == 2:
            states_one_action = states_one_action.unsqueeze(0)
        action_prob = torch.zeros(action_num)
        action_prob[action_type] = 1
        data[action_prob] = states_one_action
    return data


def comb_buffers(states, actions, rewards):
    assert len(states) == len(actions) == len(rewards)
    data = []
    for i in range(len(states)):
        data.append([states[i].unsqueeze(0), actions[i], rewards[i]])
    return data


def split_data_by_reward(states, actions, reward, action_num):
    reward_types = torch.unique(reward)
    data = {}

    for reward_type in reward_types:
        reward_type_indices = reward == reward_type
        reward_type_states = states[reward_type_indices]
        reward_type_actions = actions[reward_type_indices]
        data[reward_type] = split_data_by_action(reward_type_states, reward_type_actions, action_num)
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


def mask_name_from_state(state, obj_names, splitter):
    mask_name = ""
    for obj_i, obj_name in enumerate(obj_names):
        if state[:, obj_i, obj_i] > 0:
            mask_name += f"exist_{obj_name}"
        else:
            mask_name += f"not_exist_{obj_name}"
        mask_name += splitter
    mask_name = mask_name[:-1]
    return mask_name


def all_mask_tensors(obj_names):
    mask_tensors = []
    name_combs = get_all_subsets(obj_names)
    for name_comb in name_combs:
        mask_tensor = torch.zeros(len(obj_names), dtype=torch.bool)
        for i in range(len(obj_names)):
            if obj_names[i] in name_comb:
                mask_tensor[i] = True

        mask_tensors.append(mask_tensor)

    return mask_tensors


def all_pred_tensors(all_preds):
    pred_tensors = []
    pred_combs = get_all_subsets(all_preds, empty_set=False)
    for pred_comb in pred_combs:
        pred_tensor = torch.zeros(len(all_preds), dtype=torch.bool)
        for i in range(len(all_preds)):
            if all_preds[i] in pred_comb:
                pred_tensor[i] = True
        if pred_tensor[0] and pred_tensor[1]:
            continue
        pred_tensors.append(pred_tensor)

    return pred_tensors


def all_mask_names(obj_names):
    # states that following different masks
    masks = []
    switches = get_all_subsets(obj_names)
    for switch in switches:
        mask_name = gen_mask_name(switch, obj_names)
        masks.append(mask_name)

    return masks


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
        if torch.sum(switch_mask) > 0:
            print(f'mask: {mask_name}, number: {torch.sum(switch_mask)}')

    return masks


def mask_name_to_tensor(mask_name, splitter):
    obj_existences = mask_name.split(splitter)
    existence = []
    for obj in obj_existences:
        if "not" in obj:
            existence.append(False)
        else:
            existence.append(True)

    return existence


def check_pred_satisfaction(states, all_preds, mask, objs, prop_indices, p_spaces=None, mode="fit"):
    # filter out states following the same mask as current behavior
    if mask is not None:
        obj_A = states[mask, objs[0]]
        obj_B = states[mask, objs[1]]
    else:
        obj_A = states[:, objs[0]]
        obj_B = states[:, objs[1]]

    sample_num = len(obj_A)
    data_A = obj_A[:, prop_indices].unsqueeze(-1).repeat(1, 1, int(len(all_preds) / len(prop_indices))).reshape(-1,
                                                                                                                len(all_preds))
    data_B = (obj_B[:, prop_indices].unsqueeze(-1).repeat(1, 1, int(len(all_preds) / len(prop_indices))).reshape(-1,
                                                                                                                 len(all_preds)))
    p_satisfactions = []
    for p_i, pred in enumerate(all_preds):
        if mode == "fit":
            p_satisfaction = pred.fit(data_A[:, p_i], data_B[:, p_i], objs)
        elif mode == "eval":
            p_satisfaction, p_values = pred.eval(data_A[:, p_i], data_B[:, p_i], p_spaces[p_i])
        else:
            raise ValueError
        p_satisfactions.append(p_satisfaction)

    return p_satisfactions, sample_num


def arrange_mps(combs_obj_type, prop_combs, mask_combs, obj_dict, pred_satisfactions):
    # get all obj combinations
    key_nums = 0
    combs_obj, keys_obj_combs, key_nums = get_obj_combs(combs_obj_type, obj_dict, key_nums)
    combs_obj_pred, keys_obj_pred_combs, key_nums = get_obj_pred_combs(combs_obj, pred_satisfactions, keys_obj_combs,
                                                                       key_nums)
    combs_obj_pred_prop, keys_obj_pred_prop_combs, key_nums = get_obj_pred_prop_combs(combs_obj_pred, prop_combs,
                                                                                      keys_obj_pred_combs, key_nums)

    combs_obj_prop_pred_mask, keys_obj_prop_pred_mask_combs, key_nums = get_obj_prop_pred_mask_combs(
        combs_obj_pred_prop, keys_obj_pred_prop_combs, mask_combs, key_nums)
    return combs_obj_prop_pred_mask, keys_obj_prop_pred_mask_combs, key_nums


def get_obj_combs(obj_type_combs, obj_dict, key_nums):
    obj_types = list(obj_dict.keys())
    obj_combs = []
    for obj_type_comb in obj_type_combs:
        obj_type1 = obj_dict[obj_types[obj_type_comb[0]]]
        obj_type2 = obj_dict[obj_types[obj_type_comb[1]]]
        for obj_1_id in obj_type1:
            for obj_2_id in obj_type2:
                obj_combs.append([obj_1_id, obj_2_id])

    key_dict = {"obj": list(range(key_nums, key_nums + len(obj_type_combs[0])))}
    key_nums_update = key_nums + len(obj_type_combs[0])
    return obj_combs, key_dict, key_nums_update


def get_obj_pred_prop_combs(obj_pred_combs, prop_codes, obj_combs_keys, key_nums):
    obj_prop_combs = []
    for obj_pred_comb in obj_pred_combs:
        obj_prop_combs.append(obj_pred_comb + prop_codes)
    new_key_nums = len(prop_codes)
    obj_combs_keys["prop"] = list(range(key_nums, key_nums + new_key_nums))
    key_nums_update = key_nums + new_key_nums

    return obj_prop_combs, obj_combs_keys, key_nums_update


def get_obj_pred_combs(obj_combs, pred_satisfactions, obj_prop_combs_keys, key_nums):
    obj_prop_pred_combs = []
    for o_i, obj_comb in enumerate(obj_combs):
        obj_prop_pred_combs.append(obj_comb + pred_satisfactions[o_i])

    new_key_nums = len(pred_satisfactions[0])
    obj_prop_combs_keys["pred"] = list(range(key_nums, key_nums + new_key_nums))
    key_nums_update = key_nums + new_key_nums
    return obj_prop_pred_combs, obj_prop_combs_keys, key_nums_update


def get_obj_prop_pred_mask_combs(combs_obj_pred_prop, keys_obj_pred_prop_combs, mask_combs, key_nums):
    obj_prop_pred_mask_combs = []
    for obj_prop_comb in combs_obj_pred_prop:
        obj_prop_pred_mask_combs.append(obj_prop_comb + mask_combs)
    new_key_nums = len(mask_combs)
    keys_obj_pred_prop_combs["mask"] = list(range(key_nums, key_nums + new_key_nums))
    key_nums_update = key_nums + new_key_nums

    return obj_prop_pred_mask_combs, keys_obj_pred_prop_combs, key_nums_update


def get_obj_type_existence(data, obj_type_indices, obj_type_names):
    # mask satisfaction of given data
    obj_type_existences = []
    for obj_type, obj_indices in obj_type_indices.items():
        obj_type_prop_idx = obj_type_names.index(obj_type)
        obj_type_existence = False
        for obj_idx in obj_indices:
            if data[0, obj_idx, obj_type_prop_idx] > 0:
                obj_type_existence = True
        obj_type_existences.append(obj_type_existence)
    return obj_type_existences


def oppm_eval(data, oppm_comb, oppm_keys, preds, p_spaces, obj_type_indices, obj_type_names):
    id_objs = [oppm_comb[idx] for idx in oppm_keys["obj"]]
    id_props = [oppm_comb[idx] for idx in oppm_keys["prop"]]
    satisfy_predicates = [oppm_comb[idx] for idx in oppm_keys["pred"]]
    satisfy_masks = [oppm_comb[idx] for idx in oppm_keys["mask"]]

    data_A = data[:, id_objs[0], id_props]
    data_B = data[:, id_objs[1], id_props]

    data_As = data_A.repeat(1, int(len(preds) / len(id_props)))
    data_Bs = data_B.repeat(1, int(len(preds) / len(id_props)))

    # predicates satisfaction of given data
    state_predicates = []
    p_spaces = p_spaces * len(id_props)

    for p_i, pred in enumerate(preds):
        satisfy = pred.eval_batch(data_As[:, p_i], data_Bs[:, p_i], p_spaces[p_i])
        state_predicates.append(satisfy)

    # mask satisfaction of given data
    state_masks = get_obj_type_existence(data, obj_type_indices, obj_type_names)

    pred_satisfaction = satisfy_predicates == state_predicates
    mask_satisfaction = satisfy_masks == state_masks

    satisfaction = False
    if pred_satisfaction and mask_satisfaction:
        return True

    return satisfaction


def get_smp_facts(types, relate_2_obj_types, relate_2_prop_types, all_preds):
    facts = []

    masks = all_mask_names(types)
    pred_combs = all_pred_tensors(all_preds)
    for type_comb in relate_2_obj_types:
        for mask in masks:
            for prop_types in relate_2_prop_types:
                for pred_comb in pred_combs:
                    facts.append({"mask": mask, "objs": type_comb, "props": prop_types, "pred_tensors": pred_comb,
                                  "preds": all_preds})

    return facts


def is_trivial(mask, objs):
    mask_tensor = mask_name_to_tensor(mask, config.mask_splitter)
    for obj in objs:
        if not mask_tensor[obj]:
            return True
    return False

def satisfy_fact(fact, states, mask_dict):
    mask = mask_dict[fact["mask"]]
    objs = fact["objs"]
    props = fact["props"]
    pred_fact = fact["pred_tensors"]
    preds = fact["preds"]

    if is_trivial(fact["mask"], objs):
        return False

    obj_A = states[mask, objs[0]]
    obj_B = states[mask, objs[1]]
    if len(obj_A) == 0:
        return False
    data_A = obj_A[:, props]
    data_B = obj_B[:, props]

    pred_states = torch.zeros(pred_fact.size(), dtype=torch.bool)
    for i in range(len(preds)):
        pred_states[i] = preds[i].fit(data_A, data_B, objs)

    pred_satisfy = torch.equal(pred_states, pred_fact)
    return pred_satisfy


def update_pred_parameters(preds, action_states, behavior):
    # predict actions for each state using given behavior
    # all the params are valid
    satisfactions, params = behavior.update_pred_params(preds, action_states)

    # for the states return not satisfaction, the params have to be added to the p_space
    params[~satisfactions]