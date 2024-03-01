# Created by jing at 23.12.23
import os
import torch
import numpy as np
from tqdm import tqdm
from itertools import combinations
import itertools

from src import config
from pi import predicate
from pi.utils import math_utils, draw_utils


def get_param_range(min, max, unit):
    length = (max - min) // unit
    if length == 0:
        return torch.zeros(1)

    space = torch.zeros(int(length))
    for v_i in range(len(space)):
        space[v_i] = min + unit * v_i
    return space


def get_all_2_combinations(game_info, reverse=True):
    if reverse:
        all_combinations = [[i_1, i_2] for i_1, s_1 in enumerate(game_info) for i_2, s_2 in
                            enumerate(game_info) if i_1 != i_2]
    else:
        all_combinations = [[i_1, i_2] for i_1, s_1 in enumerate(game_info) for i_2, s_2 in
                            enumerate(game_info) if i_2 > i_1]
    return all_combinations


def get_obj_prop_combs(game_info, prop_indices):
    obj_combs = get_all_2_combinations(game_info, reverse=False)
    all_combs = []
    for prop in prop_indices:
        for obj_comb in obj_combs:
            all_combs.append(obj_comb + [prop])
    return all_combs


def get_all_facts(game_info, prop_indices):
    obj_types = get_all_2_combinations(game_info)
    prop_types = [[each] for each in prop_indices]
    obj_names = [info["name"] for info in game_info]
    facts = []
    type_combs = list(itertools.product(obj_types, prop_types))
    for type_comb in type_combs:
        masks = exist_mask_names(obj_names, type_comb[0])
        for mask in masks:
            mask_tensor = mask_name_to_tensor(mask, config.mask_splitter)
            facts.append({"mask": mask_tensor.tolist(),
                          "objs": type_comb[0],
                          "props": type_comb[1],
                          "preds": [0]
                          })
    print(f"- Number of facts: {len(facts)}")
    return facts


def get_fact_obj_combs(game_info, fact):
    type_0_index = fact["objs"][0]
    type_1_index = fact["objs"][1]
    obj_0_indices = game_info[type_0_index]["indices"]
    obj_1_indices = game_info[type_1_index]["indices"]
    obj_combs = enumerate_two_combs(obj_0_indices, obj_1_indices)
    return obj_combs


def get_states_delta(states, obj_a, obj_b, prop):
    delta_satisfaction = torch.zeros((states.size(0)), dtype=torch.bool)
    for s_i in range(1, len(states)):
        dist_past = torch.abs(states[s_i - 1][obj_a][prop] - states[s_i - 1][obj_b][prop])
        dist_current = torch.abs(states[s_i][obj_a][prop] - states[s_i][obj_b][prop])
        if dist_past > dist_current:
            delta_satisfaction[s_i] = True

    # first delta equal to the second delta
    delta_satisfaction[0] = delta_satisfaction[1]
    return delta_satisfaction


def fact_is_true(states, fact, game_info):
    prop = fact["props"]
    obj_combs = torch.tensor(get_fact_obj_combs(game_info, fact))
    fact_mask = torch.repeat_interleave(torch.tensor(fact["mask"]).unsqueeze(0), len(states), 0)
    state_mask = mask_tensors_from_states(states, game_info)
    fact_satisfaction = (fact_mask == state_mask).prod(dim=-1).bool()
    pred_satisfaction = torch.zeros(len(states), dtype=torch.bool)
    pred = predicate.GT_Closest()

    obj_a_indices = obj_combs[:, 0].unique()
    obj_b_indices = obj_combs[:, 1].unique()
    if len(obj_a_indices) == 1:
        data_A = states[:, obj_a_indices, prop]
        data_B = states[:, obj_b_indices, prop]
        pred_satisfaction = pred.eval(data_A, data_B)

    fact_satisfaction *= pred_satisfaction
    assert len(fact_satisfaction) == len(states)
    return fact_satisfaction


def check_fact_truth(facts, states, actions, game_info):
    truth_table = torch.zeros((len(states), len(facts)), dtype=torch.bool)
    for f_i in tqdm(range(len(facts)), ascii=True, desc=f"Fact table"):
        fact_satisfaction = fact_is_true(states, facts[f_i], game_info)
        truth_table[:, f_i] = fact_satisfaction
    assert truth_table.sum() > 0
    return truth_table


def remove_trivial_facts(facts, fact_actions):
    trivial_indices = fact_actions.sum(dim=-1) == len(facts[0]["pred_tensors"])
    trivial_indices += fact_actions.sum(dim=-1) == 0
    non_trivial_facts = facts[~trivial_indices]
    return non_trivial_facts


def extend_one_fact_to_fact(facts, fact_actions, base_facts, base_fact_actions):
    new_facts = []
    for f_i in range(len(facts) - 1):
        fa_i = fact_actions[f_i]
        for f_j in range(f_i + 1, len(fact_actions)):
            fa_j = fact_actions[f_j]
            if (fa_i * fa_j).sum() > 0:
                new_facts.append([base_facts[f_i], base_facts[f_j]])
    return new_facts


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
        try:
            states_one_action = states[index_action]
        except IndexError:
            print("Index error")
        states_one_action = states_one_action.squeeze()
        if len(states_one_action.shape) == 2:
            states_one_action = states_one_action.unsqueeze(0)
        action_prob = torch.zeros(action_num)
        action_prob[action_type] = 1
        data[action_prob] = states_one_action
    return data


def comb_buffers(states, actions, rewards):
    data = []

    for game_i in range(len(states)):
        game_data = []
        for s_i in range(len(states[game_i])):
            game_data.append([states[game_i][s_i].unsqueeze(0), actions[game_i][s_i], rewards[game_i][s_i]])
        data.append(game_data)
    return data


def split_data_by_reward(states, rewards, actions, zero_reward, game_info):
    mask_neg_reward = rewards < zero_reward
    mask_zero_reward = rewards == zero_reward
    mask_pos_reward = rewards > zero_reward

    neg_states = states[mask_neg_reward]
    neg_rewards = rewards[mask_neg_reward]
    neg_actions = actions[mask_neg_reward]
    neg_masks = mask_tensors_from_states(neg_states, game_info)

    pos_states = states[mask_pos_reward]
    pos_rewards = rewards[mask_pos_reward]
    pos_actions = actions[mask_pos_reward]
    pos_masks = mask_tensors_from_states(pos_states, game_info)

    zero_states = states[mask_zero_reward]
    zero_rewards = rewards[mask_zero_reward]
    zero_actions = actions[mask_zero_reward]
    zero_masks = mask_tensors_from_states(zero_states, game_info)

    pos_data = {"states": pos_states, "actions": pos_actions, "rewards": pos_rewards, "masks": pos_masks}
    neg_data = {"states": neg_states, "actions": neg_actions, "rewards": neg_rewards, "masks": neg_masks}
    zero_data = {"states": zero_states, "actions": zero_actions, "rewards": zero_rewards, "masks": zero_masks}
    return pos_data, neg_data, zero_data


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


def mask_name_from_state(state, game_info, splitter):
    mask_name = ""

    for obj_name, obj_indices, prop_obj_exist in game_info:
        if state[:, obj_indices, prop_obj_exist].sum() > 0:
            mask_name += f"exist_{obj_name}"
        else:
            mask_name += f"not_exist_{obj_name}"
        mask_name += splitter
    mask_name = mask_name[:-1]
    return mask_name


def mask_tensors_from_states(states, obj_info):
    mask_tensors = torch.zeros((len(states), len(obj_info)), dtype=torch.bool)
    for i in range(len(obj_info)):
        name, obj_indices = obj_info[i]['name'], obj_info[i]["indices"]
        obj_exist_counter = states[:, obj_indices, i].sum(dim=-1)
        mask_tensors[:, i] = obj_exist_counter > 0

    mask_tensors = mask_tensors.bool()
    return mask_tensors


def mask_tensors_from_series_states(states, obj_info):
    mask_tensors = torch.zeros((states.shape[0], states.shape[1], len(obj_info)), dtype=torch.bool)
    for i in range(len(obj_info)):
        name, obj_indices = obj_info[i]['name'], obj_info[i]["indices"]

        obj_exist_counter = states[:, 0, obj_indices, i].sum(dim=-1)
        mask_tensors[:, :, i] = torch.repeat_interleave((obj_exist_counter > 0).unsqueeze(1), mask_tensors.shape[1],
                                                        dim=1)

    mask_tensors = mask_tensors.bool()
    return mask_tensors


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


def exist_mask_names(obj_names, exist_indices):
    # states that following different masks
    uncertain_obj_names = [obj_names[i] for i in range(len(obj_names)) if i not in exist_indices]
    masks = []
    switches = get_all_subsets(uncertain_obj_names)
    for switch_names in switches:
        switch_names += [obj_names[i] for i in exist_indices]
        mask_name = gen_mask_name(switch_names, obj_names)
        masks.append(mask_name)

    return masks


def all_exist_mask(states, game_info):
    obj_names = [name for name, _, _ in game_info]
    obj_masks = {}
    for name, obj_indices, prop_index in game_info:
        obj_exist = states[:, obj_indices, prop_index].sum(dim=-1) > 0
        obj_masks[f'{name}'] = obj_exist

    # states that following different masks
    masks = {}
    obj_type_combs = get_all_subsets(obj_names)
    for obj_type_comb in obj_type_combs:
        switch_masks = []
        for name, obj_indices, prop_index in game_info:
            mask = ~obj_masks[name]
            if name in obj_type_comb:
                mask = ~mask
            switch_masks.append(mask.tolist())
        switch_mask = torch.prod(torch.tensor(switch_masks).float(), dim=0).bool()
        mask_name = gen_mask_name(obj_type_comb, obj_names)
        masks[mask_name] = switch_mask
        # if torch.sum(switch_mask) > 0:
        #     print(f'mask: {mask_name}, number: {torch.sum(switch_mask)}')

    return masks


def mask_name_to_tensor(mask_name, splitter):
    obj_existences = mask_name.split(splitter)
    existence = []
    for obj in obj_existences:
        if "not" in obj:
            existence.append(False)
        else:
            existence.append(True)

    return torch.tensor(existence)


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


def all_pred_combs(pred_lists):
    pred_combs = []
    for dir_i in range(len(pred_lists[0])):
        pred_combs.append([pred_lists[0][dir_i]])
    return pred_combs


def get_smp_facts(game_info, relate_2_obj_types, relate_2_prop_types):
    # at_least_preds = predicate.get_at_least_preds(1, config.dist_num, config.max_dist)
    # at_most_preds = predicate.get_at_most_preds(1, config.dist_num, config.max_dist)
    facts = []
    obj_names = [name for name, _, _ in game_info]
    for type_comb in relate_2_obj_types:
        for delta_dist in [True, False]:
            masks = exist_mask_names(obj_names, type_comb)
            for mask in masks:
                for prop_types in relate_2_prop_types:
                    facts.append({"mask": mask,
                                  "objs": type_comb,
                                  "props": prop_types,
                                  "preds": [0],
                                  "delta": delta_dist
                                  })

    return facts


def is_trivial(mask, objs):
    mask_tensor = mask_name_to_tensor(mask, config.mask_splitter)
    for obj in objs:
        if not mask_tensor[obj]:
            return True
    return False


def enumerate_two_combs(list_A, list_B):
    combs = []
    for a in list_A:
        for b in list_B:
            combs.append((a, b))
    return combs


def get_obj_type_combs(game_info, fact):
    objs = fact["objs"]
    _, obj_A_indices, _ = game_info[objs[0]]
    _, obj_B_indices, _ = game_info[objs[1]]
    obj_combs = enumerate_two_combs(obj_A_indices, obj_B_indices)

    return obj_combs


def satisfy_fact(fact, states, mask_dict, game_info, eval_mode=False):
    fact_mask = mask_dict[fact["mask"]]
    objs = fact["objs"]
    props = fact["props"]
    pred_fact = fact["pred_tensors"]
    preds = fact["preds"]

    if is_trivial(fact["mask"], objs):
        return False

    obj_combs = get_obj_type_combs(game_info, fact)
    # fact is true if at least one comb is true

    for obj_comb in obj_combs:
        obj_A = states[fact_mask, obj_comb[0]]
        obj_B = states[fact_mask, obj_comb[1]]
        if len(obj_A) < 10:
            return False
        data_A = obj_A[:, props]
        data_B = obj_B[:, props]

        pred_states = torch.zeros(pred_fact.size(), dtype=torch.bool)
        for i in range(len(preds)):
            if eval_mode:
                try:
                    pred_satisfactions = preds[i].eval(data_A, data_B, objs)

                    # if len(data_A) > 1:
                    #     satisfy_percent = pred_satisfactions.sum()/len(pred_satisfactions)
                    #     if satisfy_percent > 0.9:
                    #         pred_satisfactions = True
                    #     else:
                    #         pred_satisfactions = False

                    pred_states[i] = pred_satisfactions
                except RuntimeError:
                    print("watch")
                    preds[i].eval(data_A, data_B, objs)
            else:
                pred_states[i] = preds[i].fit(data_A, data_B, objs)

        pred_satisfy = torch.equal(pred_states, pred_fact)

        # expanding parameters
        if pred_satisfy:
            # for i in range(len(preds)):
            #     if pred_fact[i]:
            #         preds[i].expand_space(data_A, data_B)

            return True

    return False


def update_pred_parameters(preds, action_states, behaviors, game_info):
    # predict actions for each state using given behavior
    # all the params are valid

    for behavior in behaviors:
        obj_combs = get_obj_type_combs(game_info, behavior.fact)

        for action, state_actions in action_states.items():
            if not torch.equal(behavior.action, action):
                behavior_satisfaction, _ = behavior.eval_behavior(preds, state_actions, game_info)

                # wrong satisfied states
                wrong_satisfied_states = state_actions[behavior_satisfaction]

                for p_i in range(len(preds)):
                    if behavior.fact["pred_tensors"][p_i]:
                        for obj_comb in obj_combs:
                            obj_A = wrong_satisfied_states[:, obj_comb[0]]
                            obj_B = wrong_satisfied_states[:, obj_comb[1]]
                            data_A = obj_A[:, behavior.fact['props']]
                            data_B = obj_B[:, behavior.fact['props']]
                            preds[p_i].refine_space(data_A, data_B)


def search_behavior_conflict_states(behavior, data_combs, game_info, action_num):
    conflict_indices = torch.zeros(len(data_combs), dtype=torch.bool)
    for d_i in range(len(data_combs)):
        state, action, reward, reason_resource = data_combs[d_i]

        # if action == 2 and state[0, 0, 4] > state[0, 1, 4] and state[0, 1, 1] > 0.6:
        #     print(f"agent enemy distance: ")
        #     print(f"agent key distance: ")
        # all facts have to be matched
        fact_match = behavior.eval_behavior(state, game_info)
        if not fact_match:
            conflict_indices[d_i] = False
        if fact_match:
            # if action == 2:
            #     print("watch")
            action_not_match = behavior.action.argmax() != action
            conflict_indices[d_i] = action_not_match

    # beh_not_match_states = [data_combs[i][0].tolist() for i in range(len(conflict_indices)) if conflict_indices[i]]
    # beh_not_match_state_actions = [data_combs[i][1].tolist() for i in range(len(conflict_indices)) if
    #                                conflict_indices[i]]
    # beh_not_match_states = torch.tensor(beh_not_match_states).squeeze()
    # beh_not_match_state_actions = torch.tensor(beh_not_match_state_actions)

    conflict_percent = sum(conflict_indices) / len(data_combs)
    # print(f"\n- Teacher Behavior: "
    #       f"{teacher_behavior.clause}, "
    #       f"conflict indices: {sum(conflict_indices)}/{len(self.data_combs)}, "
    #       f"conflict percent: {conflict_percent}\n")

    satisfied_data, conflict_data = prepare_student_data(conflict_indices, data_combs, action_num)

    return satisfied_data, conflict_data


def prepare_student_data(conflict_indices, data_combs, action_num):
    beh_not_match_states = [data_combs[i][0].tolist() for i in range(len(conflict_indices)) if conflict_indices[i]]
    beh_not_match_states = torch.tensor(beh_not_match_states).squeeze()
    if conflict_indices.sum() < 2:
        return data_combs, None
    beh_match_states = [data_combs[i][0].tolist() for i in range(len(conflict_indices)) if not conflict_indices[i]]
    beh_match_states = torch.tensor(beh_match_states).squeeze()

    beh_not_match_state_actions = [data_combs[i][1].tolist() for i in range(len(conflict_indices)) if
                                   conflict_indices[i]]
    beh_not_match_state_actions = torch.tensor(beh_not_match_state_actions)

    beh_match_state_actions = [data_combs[i][1].tolist() for i in range(len(conflict_indices)) if
                               not conflict_indices[i]]
    beh_match_state_actions = torch.tensor(beh_match_state_actions)

    conflict_data = split_data_by_action(beh_not_match_states, beh_not_match_state_actions, action_num)

    zero_rewards = torch.zeros(len(beh_match_states))
    zero_reason_source = torch.zeros(len(beh_match_states))
    satisfied_data = comb_buffers(beh_match_states, beh_match_state_actions, zero_rewards, zero_reason_source)

    return satisfied_data, conflict_data


def back_check(data, conflict_behaviors, game_info, action_num):
    behaviors = []
    for c_beh in conflict_behaviors:
        satisfied_data, conflict_data = search_behavior_conflict_states(c_beh, data, game_info, action_num)
        print(f"student behavior: {c_beh.clause}, satisfy percent: {len(satisfied_data)}/{len(data)}")
        if len(satisfied_data) / len(data) > 0.999:
            behaviors.append(c_beh)
    return behaviors


def get_fact_combs(comb_iteration, total):
    comb = torch.tensor(list(combinations(list(range(total)), comb_iteration)))
    # print(f"fact combinations: {len(comb)}")
    return comb


def fact_grouping_by_head(facts):
    head_id = -1
    checked_head = []
    fact_head_ids = []
    fact_bodies = []
    fact_heads = []
    for f_i in range(len(facts)):
        head = facts[f_i]["mask"] + str(facts[f_i]["objs"]) + str(facts[f_i]["props"])
        body = {"min": facts[f_i]["preds"][1].p_bound["min"], "max": facts[f_i]["preds"][2].p_bound["max"]}
        if head not in checked_head:
            checked_head.append(head)
            head_id += 1
        fact_head_ids.append(head_id)
        fact_bodies.append(body)
        fact_heads.append(head)
    fact_head_ids = torch.tensor(fact_head_ids)
    return checked_head, fact_head_ids, fact_bodies, fact_heads


def get_activities(buffer, activity_size):
    pass


def save_truth_tables(truth_tables, filename):
    torch.save(truth_tables, filename)


def stat_facts_in_states(fact_combs, fact_num, fact_table, fact_anti_table, rewards):
    fact_pos_num = torch.zeros(fact_combs.size(0))
    fact_neg_num = torch.zeros(fact_combs.size(0))
    pos_states = torch.zeros((len(fact_combs), fact_table.size(0)), dtype=torch.bool)
    neg_states = torch.zeros((len(fact_combs), fact_anti_table.size(0)), dtype=torch.bool)
    fact_rewards = torch.zeros(fact_combs.size(0))
    for ci_i in tqdm(range(len(fact_combs)), ascii=True, desc=f"{fact_num + 1} fact behavior search"):
        fact_comb = fact_combs[ci_i]
        pos_states[ci_i] = fact_table[:, fact_comb].prod(dim=-1).bool()
        neg_states[ci_i] = fact_anti_table[:, fact_comb].prod(dim=-1).bool()
        fact_pos_num[ci_i] = pos_states[ci_i].sum()
        fact_neg_num[ci_i] = neg_states[ci_i].sum()
        fact_rewards[ci_i] = rewards[pos_states[ci_i]].median()

    return fact_pos_num, fact_neg_num, pos_states, neg_states, fact_rewards


def get_state_delta(state, state_last, obj_comb):
    dist = torch.abs(state[obj_comb[0]] - state[obj_comb[1]])
    dist_past = torch.abs(state_last[obj_comb[0]] - state_last[obj_comb[1]])
    if dist_past > dist:
        delta = True
    else:
        delta = False

    return delta


def stat_zero_rewards(states, actions, rewards, zero_reward, game_info, prop_indices, var_th, stat_type, action_names,
                      step_dist):
    import numpy as np
    cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

    threshold_percentile = np.percentile(cumulative_avg, 50)
    cumulative_avg_thresholded = np.where(cumulative_avg < threshold_percentile, 0, cumulative_avg)
    mask_reward = cumulative_avg_thresholded > 0
    rewards_pos = rewards[mask_reward]
    states_pos = states[mask_reward]
    actions_pos = actions[mask_reward]
    masks_pos = mask_tensors_from_states(states_pos, game_info)

    states_neg = states[~mask_reward]
    actions_neg = actions[~mask_reward]
    masks_neg = mask_tensors_from_states(states_neg, game_info)

    mask_pos_types = masks_pos.unique(dim=0)
    action_pos_types = actions_pos.unique()
    obj_types = get_all_2_combinations(game_info, reverse=False)
    prop_types = [prop_indices]
    type_combs = list(itertools.product(mask_pos_types, action_pos_types, obj_types, prop_types))
    states_stats = []
    variances = torch.zeros(len(type_combs))
    variances_neg = torch.zeros(len(type_combs))
    means = torch.zeros(len(type_combs))
    means_neg = torch.zeros(len(type_combs))
    percentage = torch.zeros(len(type_combs))
    for t_i in range(len(type_combs)):
        mask_type, action_type, obj_type, prop_type = type_combs[t_i]
        if mask_type[obj_type].prod() == False:
            variances[t_i] = 1e+20
            means[t_i] = 1e+20
            variances_neg[t_i] = 1e+20
            means_neg[t_i] = 1e+20
            states_stats.append([])
            continue
        # print(type_combs[t_i])
        mask_action_state_pos = ((actions_pos == action_type) * (masks_pos == mask_type).prod(-1).bool())
        mask_action_state_neg = ((actions_neg == action_type) * (masks_neg == mask_type).prod(-1).bool())
        states_action_pos = states_pos[mask_action_state_pos]
        states_action_neg = states_neg[mask_action_state_neg]

        obj_0_indices = game_info[obj_type[0]]["indices"]
        obj_1_indices = game_info[obj_type[1]]["indices"]
        obj_combs = torch.tensor(list(itertools.product(obj_0_indices, obj_1_indices)))
        obj_a_indices = obj_combs[:, 0].unique()
        obj_b_indices = obj_combs[:, 1].unique()
        if len(obj_a_indices) == 1 and states_action_pos.shape[0] > 8 and states_action_neg.shape[0] > 20:
            action_name = action_names[action_type]
            action_dir = math_utils.action_to_deg(action_name)
            data_A = states_action_pos[:, obj_a_indices][:, :, prop_type]
            data_B = states_action_pos[:, obj_b_indices][:, :, prop_type]
            data_A_neg = states_action_neg[:, obj_a_indices][:, :, prop_type]
            data_B_neg = states_action_neg[:, obj_b_indices][:, :, prop_type]

            if "fire" in action_name or "noop" in action_name:
                data_A_one_step_move = data_A
                data_A_neg_one_step_move = data_A_neg
            else:
                data_A_one_step_move = math_utils.one_step_move(data_A, action_dir, step_dist)
                data_A_neg_one_step_move = math_utils.one_step_move(data_A_neg, action_dir, step_dist)

            dist, b_index = math_utils.dist_a_and_b_closest(data_A_one_step_move, data_B)
            dir_ab = math_utils.dir_ab_batch(data_A_one_step_move, data_B, b_index)
            dist_neg, b_neg_index = math_utils.dist_a_and_b_closest(data_A_neg_one_step_move, data_B_neg)
            dir_ab_neg = math_utils.dir_ab_batch(data_A_neg_one_step_move, data_B_neg, b_neg_index)


        else:
            dist = torch.zeros(2)
            data_A = torch.zeros(2)
            data_A_neg = torch.zeros(2)
            dist_neg = torch.zeros(2)
            dir_ab = torch.zeros(2)
            dir_ab_neg = torch.zeros(2)
        var_pos, mean_pos = torch.var_mean(dir_ab, dim=0)
        var_neg, mean_neg = torch.var_mean(dir_ab_neg, dim=0)

        var_pos = var_pos.sum()
        mean_pos = mean_pos.sum()
        var_neg = var_neg.sum()
        mean_neg = means_neg.sum()

        if len(dir_ab) < 3 or var_neg == 0 or dist.sum() == 0:
            var_pos = 1e+20
            mean_pos = 1e+20
        variances[t_i] = var_pos
        means[t_i] = mean_pos
        variances_neg[t_i] = var_neg
        means_neg[t_i] = mean_neg
        percentage[t_i] = len(states_action_pos) / len(states_pos)
        states_stats.append(
            {"dists_pos": dist, "dir_pos": dir_ab,
             "dists_neg": dist_neg, "dir_ab_neg": dir_ab_neg,
             "means": mean_pos, "variances": var_pos,
             "position_pos": data_A.squeeze(), "position_neg": data_A_neg.squeeze(),
             "action_type": action_type, "mask_type": mask_type, "prop_type": prop_type, "obj_types": obj_type,
             "indices": mask_action_state_pos})
    variances_ranked, v_rank = variances.sort()

    passed_variances = variances_ranked < var_th
    passed_comb_indices = v_rank[passed_variances]
    passed_stats = [states_stats[s_i] for s_i in passed_comb_indices]
    passed_combs = [type_combs[s_i] for s_i in passed_comb_indices]
    behs = []
    for state_stat in passed_stats:
        indices = state_stat["indices"]
        dist_range = math_utils.get_90_percent_range_2d(state_stat["dists_pos"].numpy())
        dist_one_percent_value = math_utils.closest_one_percent(state_stat["dists_pos"])
        x_one_percent_types, x_counts = dist_one_percent_value[:, 0].unique(return_counts=True)
        y_one_percent_types, y_counts = dist_one_percent_value[:, 1].unique(return_counts=True)
        x_conf = x_one_percent_types / x_one_percent_types.sum()
        y_conf = y_one_percent_types / y_one_percent_types.sum()

        dir_value = state_stat["dir_pos"]
        dir_quarter_value = math_utils.closest_quarter(dir_value)
        dir_quarter_values, dir_counts = dir_quarter_value.unique(return_counts=True)
        dir_conf = dir_counts / dir_counts.sum()
        dir_quarter_value_best = dir_quarter_values[dir_counts.argmax()].reshape(-1)
        dir_conf_best = dir_conf[dir_counts.argmax()].reshape(-1)

        behs.append({
            "x_range": x_one_percent_types.tolist(),
            "y_range": y_one_percent_types.tolist(),
            "x_conf": x_conf.tolist(),
            "y_conf": y_conf.tolist(),
            "dir_range": dir_quarter_value_best.tolist(),
            "dir_conf": dir_conf_best.tolist(),
            "dists_pos": state_stat["dists_pos"].tolist(),
            "dir_pos": state_stat["dir_pos"].tolist(),
            "position_pos": state_stat["position_pos"].tolist(),
            "position_neg": state_stat["position_neg"].tolist(),
            "dists_neg": state_stat["dists_neg"].tolist(),
            "dir_ab_neg": state_stat["dir_ab_neg"].tolist(),
            "means": state_stat["means"].tolist(),
            "variance": state_stat["variances"].tolist(),
            "action_type": state_stat["action_type"].tolist(),
            "masks": state_stat["mask_type"].tolist(),
            "obj_combs": state_stat["obj_types"],
            "prop_combs": state_stat["prop_type"],
            "rewards": rewards_pos[indices].tolist(),
        })

    return behs


def get_state_acc(states):
    state_3 = torch.cat(
        (states[:-2, :, -2:].unsqueeze(0), states[1:-1, :, -2:].unsqueeze(0), states[2:, :, -2:].unsqueeze(0)), dim=0)
    acceleration = math_utils.calculate_acceleration_2d(state_3[:, :, :, 0], state_3[:, :, :, 1]).permute(1, 2, 0)
    acceleration = math_utils.closest_one_percent(acceleration, 0.01)
    acceleration = torch.cat(
        (torch.zeros(2, acceleration.shape[1], acceleration.shape[2]).to(acceleration.device), acceleration), dim=0)
    return acceleration


def stat_o2o_rewards(win_or_lost, states, actions, rewards, game_info, prop_indices, action_names,
                     step_dist):
    dist_unit = 0.05
    def discounted_rewards(rewards, gamma=0.2):
        discounted = []
        running_add = 0
        for r in reversed(rewards):
            running_add = running_add * gamma + r
            discounted.insert(0, running_add)
        return torch.tensor(discounted)

    rewards = discounted_rewards(rewards).to(states.device)
    if win_or_lost == "win":
        mask_reward = rewards > 0
    elif win_or_lost == "lost":
        mask_reward = rewards < 0
    else:
        raise ValueError
    accelerations = get_state_acc(states)
    accelerations_pos = accelerations[mask_reward]
    rewards_pos = rewards[mask_reward]
    states_pos = states[mask_reward]
    actions_pos = actions[mask_reward]
    masks_pos = mask_tensors_from_states(states_pos, game_info).to(states.device)

    states_neg = states[~mask_reward]
    actions_neg = actions[~mask_reward]
    rewards_neg = rewards[~mask_reward]
    masks_neg = mask_tensors_from_states(states_neg, game_info).to(states.device)
    accelerations_neg = accelerations[~mask_reward]
    action_pos_types = actions_pos.unique()
    obj_types = get_all_2_combinations(game_info, reverse=False)
    obj_types = [ot for ot in obj_types if ot[0] == 0]
    prop_types = [prop_indices]
    dir_types = torch.arange(-0.75, 1.25, 0.25, dtype=torch.float).to(states.device)
    x_types = torch.arange(0, 1, dist_unit, dtype=torch.float).to(states.device)
    y_types = torch.arange(0, 1, dist_unit, dtype=torch.float).to(states.device)
    acc_dir_types = torch.arange(-0.75, 1.25, 0.25, dtype=torch.float).to(states.device)
    acc_x_types = accelerations[:, :, 0].unique()
    acc_y_types = accelerations[:, :, 1].unique()
    type_combs = list(
        itertools.product(action_pos_types, obj_types, prop_types, dir_types, x_types, y_types, acc_x_types,
                          acc_y_types, acc_dir_types))
    states_stats = []

    for t_i in tqdm(range(len(type_combs)), desc="O2O TypeComb Stat"):

        action_type, obj_type, prop_type, dir_type, x_type, y_type, acc_x_type, acc_y_type, acc_dir_type = type_combs[
            t_i]
        # print(type_combs[t_i])
        mask_obj_pos = masks_pos[:, obj_type].prod(dim=1) > 0
        mask_obj_neg = masks_neg[:, obj_type].prod(dim=1) > 0
        mask_action_state_pos = (actions_pos == action_type) * mask_obj_pos
        mask_action_state_neg = (actions_neg == action_type) * mask_obj_neg
        states_action_pos = states_pos[mask_action_state_pos]
        states_action_neg = states_neg[mask_action_state_neg]
        rewards_action_pos = rewards_pos[mask_action_state_pos]
        rewards_action_neg = rewards_neg[mask_action_state_neg]
        satisfy_state_num = 0
        obj_0_indices = game_info[obj_type[0]]["indices"]
        obj_1_indices = game_info[obj_type[1]]["indices"]
        obj_combs = torch.tensor(list(itertools.product(obj_0_indices, obj_1_indices)))
        obj_a_indices = obj_combs[:, 0].unique()
        obj_b_indices = obj_combs[:, 1].unique()
        if len(obj_a_indices) == 1 and states_action_pos.shape[0] > 0 and states_action_neg.shape[0] > 0:
            action_name = action_names[action_type]
            action_dir = math_utils.action_to_deg(action_name)
            data_A = states_action_pos[:, obj_a_indices][:, :, prop_type]
            data_B = states_action_pos[:, obj_b_indices][:, :, prop_type]
            data_A_neg = states_action_neg[:, obj_a_indices][:, :, prop_type]
            data_B_neg = states_action_neg[:, obj_b_indices][:, :, prop_type]

            if "fire" == action_name or "noop" == action_name:
                data_A_one_step_move = data_A
                data_A_neg_one_step_move = data_A_neg
            else:
                data_A_one_step_move = math_utils.one_step_move(data_A, action_dir, step_dist)
                data_A_neg_one_step_move = math_utils.one_step_move(data_A_neg, action_dir, step_dist)
            # if data_B.shape[1]>1:
            #     print("")
            # dir_ab = math_utils.dir_ab_any(data_A_one_step_move, data_B)
            dir_ab_aligned = math_utils.dir_a_and_b_with_alignment(data_A_one_step_move, data_B).to(states.device)
            dir_ab_aligned_neg = math_utils.dir_a_and_b_with_alignment(data_A_neg_one_step_move, data_B_neg).to(
                states.device)
            # dir_ab_neg = math_utils.dir_ab_any(data_A_neg_one_step_move, data_B_neg)

            dist = math_utils.dist_a_and_b(data_A_one_step_move, data_B)
            dist_neg = math_utils.dist_a_and_b(data_A_neg_one_step_move, data_B_neg)

            dist_aligned = math_utils.closest_one_percent(dist, dist_unit)
            dist_aligned_neg = math_utils.closest_one_percent(dist_neg, dist_unit)

            mask_dist = torch.logical_and(dist_aligned[:, :, 0] == x_type, dist_aligned[:, :, 1] == y_type)
            mask_dist_neg = torch.logical_and(dist_aligned_neg[:, :, 0] == x_type, dist_aligned_neg[:, :, 1] == y_type)

            mask_dir = (dir_ab_aligned == dir_type).reshape(-1, 1)
            mask_dir_aligned_neg = (dir_ab_aligned_neg == dir_type).reshape(-1, 1)

            mask = mask_dist * mask_dir
            mask_neg = mask_dist_neg * mask_dir_aligned_neg

            reward_pos_sum = rewards_action_pos[mask.prod(dim=-1).bool()].sum()
            reward_neg_sum = rewards_action_neg[mask_neg.prod(dim=-1).bool()].sum()
            satisfy_state_num = (mask.sum(dim=1) > 0).sum()
            unsatisfied_state_num = (mask_neg.sum(dim=1) > 0).sum()

            if reward_pos_sum > reward_neg_sum and reward_pos_sum > 0:
                log_text = (f"\n act:{action_names[action_type]}, "
                            f"{game_info[obj_type[0]]['name']} {game_info[obj_type[1]]['name']} "
                            f"x:{x_type:.2f}, "
                            f"y:{y_type:.2f}, "
                            f"dir:{math_utils.pol2dir_name(dir_type)}, {satisfy_state_num.item()} pos reward: {reward_pos_sum} "
                            f"neg reward: {reward_neg_sum}")
                print(log_text)
                states_stats.append(
                    {"dir_pos": dir_ab_aligned,
                     "dists_pos": dist,
                     "dists_neg": dist_neg,
                     "reward_pos": reward_pos_sum.tolist(),
                     "reward_neg": reward_neg_sum.tolist(),
                     "position_pos": data_A.squeeze(), "position_neg": data_A_neg.squeeze(),
                     "action_type": action_type, "prop_type": prop_type, "obj_types": obj_type, "dir_type": dir_type,
                     "x_type": x_type, "y_type": y_type, "indices": mask_action_state_pos})

    behs = []
    for state_stat in states_stats:
        behs.append({
            "x_type": state_stat["x_type"].tolist(),
            "y_type": state_stat["y_type"].tolist(),
            "dir_type": state_stat["dir_type"].tolist(),
            "reward": state_stat["reward_pos"] - state_stat["reward_neg"],
            "action_type": state_stat["action_type"].tolist(),
            "obj_combs": state_stat["obj_types"],
            "prop_combs": state_stat["prop_type"],
        })

    return behs


def stat_rewards(states, actions, rewards, zero_reward, game_info, prop_indices, var_th, stat_type, action_names,
                 step_dist, max_dist):
    if stat_type == "attack":
        mask_reward = rewards > zero_reward
    elif stat_type == "defense":
        mask_reward = rewards < zero_reward
    else:
        raise ValueError

    rewards_pos = rewards[mask_reward]
    states_pos = states[mask_reward]
    actions_pos = actions[mask_reward]
    masks_pos = mask_tensors_from_states(states_pos, game_info)

    states_neg = states[~mask_reward]
    actions_neg = actions[~mask_reward]
    masks_neg = mask_tensors_from_states(states_neg, game_info)

    mask_pos_types = masks_pos.unique(dim=0)
    action_pos_types = actions_pos.unique()
    obj_types = get_all_2_combinations(game_info, reverse=False)
    prop_types = [prop_indices]
    type_combs = list(itertools.product(mask_pos_types, action_pos_types, obj_types, prop_types))
    states_stats = []
    variances = torch.zeros(len(type_combs))
    variances_neg = torch.zeros(len(type_combs))
    means = torch.zeros(len(type_combs))
    means_neg = torch.zeros(len(type_combs))
    percentage = torch.zeros(len(type_combs))
    for t_i in range(len(type_combs)):
        mask_type, action_type, obj_type, prop_type = type_combs[t_i]
        if mask_type[obj_type].prod() == False:
            variances[t_i] = 1e+20
            means[t_i] = 1e+20
            variances_neg[t_i] = 1e+20
            means_neg[t_i] = 1e+20
            states_stats.append([])
            continue
        # print(type_combs[t_i])
        mask_action_state_pos = ((actions_pos == action_type) * (masks_pos == mask_type).prod(-1).bool())
        mask_action_state_neg = ((actions_neg == action_type) * (masks_neg == mask_type).prod(-1).bool())
        states_action_pos = states_pos[mask_action_state_pos]
        states_action_neg = states_neg[mask_action_state_neg]

        obj_0_indices = game_info[obj_type[0]]["indices"]
        obj_1_indices = game_info[obj_type[1]]["indices"]
        obj_combs = torch.tensor(list(itertools.product(obj_0_indices, obj_1_indices)))
        obj_a_indices = obj_combs[:, 0].unique()
        obj_b_indices = obj_combs[:, 1].unique()
        if len(obj_a_indices) == 1 and states_action_pos.shape[0] > 8:
            action_name = action_names[action_type]
            action_dir = math_utils.action_to_deg(action_name)

            data_A = states_action_pos[:, obj_a_indices][:, :, prop_type]
            data_B = states_action_pos[:, obj_b_indices][:, :, prop_type]
            data_A_neg = states_action_neg[:, obj_a_indices][:, :, prop_type]
            data_B_neg = states_action_neg[:, obj_b_indices][:, :, prop_type]

            if "fire" in action_name:
                data_A_one_step_move = data_A
                data_A_neg_one_step_move = data_A_neg
            else:
                data_A_one_step_move = math_utils.one_step_move(data_A, action_dir, step_dist)
                data_A_neg_one_step_move = math_utils.one_step_move(data_A_neg, action_dir, step_dist)

            dist, b_index = math_utils.dist_a_and_b_closest(data_A_one_step_move, data_B)
            dir_ab = math_utils.dir_ab_batch(data_A_one_step_move, data_B, b_index)
            dist_dir_pos = torch.cat((dist, dir_ab), dim=1)

            dist_neg, b_neg_index = math_utils.dist_a_and_b_closest(data_A_neg_one_step_move, data_B_neg)
            dir_ab_neg = math_utils.dir_ab_batch(data_A_neg_one_step_move, data_B_neg, b_neg_index)
            dist_dir_neg = torch.cat((dist_neg, dir_ab_neg), dim=1)

        else:
            dist = torch.zeros(2)
            data_A = torch.zeros(2)
            data_A_neg = torch.zeros(2)
            dist_neg = torch.zeros(2)
            dist_dir_pos = torch.zeros(2)
            dist_dir_neg = torch.zeros(2)
            dir_ab = torch.zeros(2)
            dir_ab_neg = torch.zeros(2)
        var_pos, mean_pos = torch.var_mean(dist_dir_pos, dim=0)
        var_neg, mean_neg = torch.var_mean(dist_dir_neg, dim=0)

        var_pos = var_pos.sum()
        mean_pos = mean_pos.sum()
        var_neg = var_neg.sum()
        mean_neg = means_neg.sum()

        if len(dist_dir_pos) < 3 or var_neg == 0 or dist.sum() == 0:
            var_pos = 1e+20
            mean_pos = 1e+20
        variances[t_i] = var_pos
        means[t_i] = mean_pos
        variances_neg[t_i] = var_neg
        means_neg[t_i] = mean_neg
        percentage[t_i] = len(states_action_pos) / len(states_pos)
        states_stats.append(
            {"dists_pos": dist, "dir_pos": dir_ab,
             "dists_neg": dist_neg, "dir_ab_neg": dir_ab_neg,
             "means": mean_pos, "variances": var_pos,
             "position_pos": data_A.squeeze(), "position_neg": data_A_neg.squeeze(),
             "action_type": action_type, "mask_type": mask_type, "prop_type": prop_type, "obj_types": obj_type,
             "indices": mask_action_state_pos})
    variances_ranked, v_rank = variances.sort()

    passed_variances = variances_ranked < var_th
    passed_comb_indices = v_rank[passed_variances]
    passed_stats = [states_stats[s_i] for s_i in passed_comb_indices]
    passed_combs = [type_combs[s_i] for s_i in passed_comb_indices]
    behs = []
    for state_stat in passed_stats:
        indices = state_stat["indices"]
        dist_range = math_utils.get_90_percent_range_2d(state_stat["dists_pos"].numpy())
        if np.abs(dist_range).max() > max_dist:
            continue

        dist_one_percent_value = math_utils.closest_one_percent(state_stat["dists_pos"])
        x_one_percent_types, x_counts = dist_one_percent_value[:, 0].unique(return_counts=True)
        y_one_percent_types, y_counts = dist_one_percent_value[:, 1].unique(return_counts=True)
        x_conf = x_one_percent_types / x_one_percent_types.sum()
        y_conf = y_one_percent_types / y_one_percent_types.sum()

        dir_value = state_stat["dir_pos"]
        dir_quarter_value = math_utils.closest_quarter(dir_value)
        dir_quarter_values, dir_counts = dir_quarter_value.unique(return_counts=True)
        dir_conf = dir_counts / dir_counts.sum()

        action_id = state_stat["action_type"].tolist()
        action_value = math_utils.action_to_deg(action_names[action_id])
        if action_value not in dir_quarter_values.tolist() and action_value != 100:
            print(f"aciton value {action_value}, data direction: {dir_quarter_values} "
                  f"variance: {state_stat['variances'].tolist()}")
            print("")

        # dir_degree = math_utils.range_to_direction(dir_range.squeeze())
        # symbolize the data further with specific direction and distance
        behs.append({
            "x_range": x_one_percent_types.tolist(),
            "y_range": y_one_percent_types.tolist(),
            "x_conf": x_conf.tolist(),
            "y_conf": y_conf.tolist(),
            "dir_range": dir_quarter_values.tolist(),
            "dir_conf": dir_conf.tolist(),
            "dists_pos": state_stat["dists_pos"].tolist(),
            "dir_pos": state_stat["dir_pos"].tolist(),
            "position_pos": state_stat["position_pos"].tolist(),
            "position_neg": state_stat["position_neg"].tolist(),
            "dists_neg": state_stat["dists_neg"].tolist(),
            "dir_ab_neg": state_stat["dir_ab_neg"].tolist(),
            "means": state_stat["means"].tolist(),
            "variance": state_stat["variances"].tolist(),
            "action_type": state_stat["action_type"].tolist(),
            "masks": state_stat["mask_type"].tolist(),
            "obj_combs": state_stat["obj_types"],
            "prop_combs": state_stat["prop_type"],
            "rewards": rewards_pos[indices].tolist(),
        })

    return behs


def indices_and_previous_k(input_list, k, value_type):
    indices = []
    values = []
    for i in range(k, len(input_list)):
        if value_type == "positive" and input_list[i] > 0:
            values.append(input_list[i - k + 1:i + 1].tolist())
            indices.append(list(range(i - k + 1, i + 1)))
        elif value_type == "non_positive" and input_list[i] <= 0:
            values.append(input_list[i - k + 1:i + 1].tolist())
            indices.append(list(range(i - k + 1, i + 1)))
    return values, indices


def stat_series_rewards(states, actions, rewards, prop_indices, args, stat_type):
    zero_reward = args.zero_reward
    game_info = args.obj_info
    var_th = args.skill_var_th
    action_name_all = args.action_names
    step_dist = args.step_dist
    max_dist = args.max_dist
    passed_stats = []
    for skill_len in range(1, args.skill_len_max):
        _, rewards_indices_pos = indices_and_previous_k(rewards, skill_len, "positive")
        states_pos = states[rewards_indices_pos]
        actions_pos = actions[rewards_indices_pos]
        masks_pos = mask_tensors_from_series_states(states_pos, game_info)

        rewards_non_pos, rewards_indices_non_pos = indices_and_previous_k(rewards, skill_len, "non_positive")
        states_neg = states[rewards_indices_non_pos]
        actions_neg = actions[rewards_indices_non_pos]
        masks_neg = mask_tensors_from_series_states(states_neg, game_info)

        mask_pos_types = masks_pos[:, 0].unique(dim=0)
        action_pos_types = actions_pos.unique(dim=0)
        obj_types = get_all_2_combinations(game_info, reverse=False)
        prop_types = [prop_indices]
        type_combs = list(itertools.product(mask_pos_types, action_pos_types, obj_types, prop_types))
        states_stats = []
        variances = torch.zeros(len(type_combs))
        variances_neg = torch.zeros(len(type_combs))
        means = torch.zeros(len(type_combs))
        means_neg = torch.zeros(len(type_combs))
        percentage = torch.zeros(len(type_combs))
        for t_i in range(len(type_combs)):
            mask_type, action_type, obj_type, prop_type = type_combs[t_i]
            if mask_type[obj_type].prod() == False:
                variances[t_i] = 1e+20
                means[t_i] = 1e+20
                variances_neg[t_i] = 1e+20
                means_neg[t_i] = 1e+20
                states_stats.append([])
                continue
            # print(type_combs[t_i])
            mask_action_state_pos = (
                    (actions_pos == action_type).prod(-1) * (masks_pos[:, 0] == mask_type).prod(-1)).bool()
            mask_action_state_neg = (
                    (actions_neg == action_type).prod(-1) * (masks_neg[:, 0] == mask_type).prod(-1)).bool()
            states_action_pos = states_pos[mask_action_state_pos]
            states_action_neg = states_neg[mask_action_state_neg]

            obj_0_indices = game_info[obj_type[0]]["indices"]
            obj_1_indices = game_info[obj_type[1]]["indices"]
            obj_combs = torch.tensor(list(itertools.product(obj_0_indices, obj_1_indices)))
            obj_a_indices = obj_combs[:, 0].unique()
            obj_b_indices = obj_combs[:, 1].unique()

            dist = torch.zeros(2)
            data_A = torch.zeros(2)
            data_A_neg = torch.zeros(2)
            dist_neg = torch.zeros(2)
            dist_dir_pos = torch.zeros(skill_len, data_A.shape[0], 3)
            dist_dir_neg = torch.zeros(skill_len, data_A_neg.shape[0], 3)
            direction = torch.zeros(2)
            direction_neg = torch.zeros(2)

            if len(obj_a_indices) == 1 and states_action_pos.shape[0] > 8 and states_action_neg.shape[0] > 8:
                action_names = [action_name_all[a_type] for a_type in action_type]
                action_dir = [math_utils.action_to_deg(a_name) for a_name in action_names]

                data_A = states_action_pos[:, :, obj_a_indices][:, :, :, prop_type]
                data_B = states_action_pos[:, :, obj_b_indices][:, :, :, prop_type]
                data_A_neg = states_action_neg[:, :, obj_a_indices][:, :, :, prop_type]
                data_B_neg = states_action_neg[:, :, obj_b_indices][:, :, :, prop_type]

                # variance calculation
                dist_dir_pos = torch.zeros(skill_len, data_A.shape[0], 3)
                dist_dir_neg = torch.zeros(skill_len, data_A_neg.shape[0], 3)
                dist = torch.zeros(skill_len, data_A.shape[0], 2)
                dist_neg = torch.zeros(skill_len, data_A_neg.shape[0], 2)
                direction = torch.zeros(skill_len, data_A.shape[0], 1)
                direction_neg = torch.zeros(skill_len, data_A_neg.shape[0], 1)

                for a_i, action_name in enumerate(action_names):
                    if "fire" in action_name:
                        data_A_one_step_move = data_A[:, a_i]
                        data_A_neg_one_step_move = data_A_neg[:, a_i]
                    else:
                        data_A_one_step_move = math_utils.one_step_move(data_A[:, a_i], action_dir[a_i], step_dist)
                        data_A_neg_one_step_move = math_utils.one_step_move(data_A_neg[:, a_i], action_dir[a_i],
                                                                            step_dist)

                    dist[a_i], b_index = math_utils.dist_a_and_b_closest(data_A_one_step_move, data_B[:, a_i])
                    direction[a_i] = math_utils.dir_ab_batch(data_A_one_step_move, data_B[:, a_i], b_index)
                    dist_dir_pos[a_i] = torch.cat((dist[a_i], direction[a_i]), dim=1)

                    dist_neg[a_i], b_neg_index = math_utils.dist_a_and_b_closest(data_A_neg_one_step_move,
                                                                                 data_B_neg[:, a_i])
                    direction_neg[a_i] = math_utils.dir_ab_batch(data_A_neg_one_step_move, data_B_neg[:, a_i],
                                                                 b_neg_index)
                    dist_dir_neg[a_i] = torch.cat((dist_neg[a_i], direction_neg[a_i]), dim=1)

            var_pos, mean_pos = torch.var_mean(dist_dir_pos, dim=1)
            var_neg, mean_neg = torch.var_mean(dist_dir_neg, dim=1)

            var_pos = var_pos.sum(dim=1).mean()
            mean_pos = mean_pos.sum(dim=1).mean()
            var_neg = var_neg.sum(dim=1).mean()
            mean_neg = mean_neg.sum(dim=1).mean()

            if dist_dir_pos.shape[1] < 3 or var_neg == 0 or dist.sum() == 0:
                var_pos = 1e+20
                mean_pos = 1e+20
            variances[t_i] = var_pos
            means[t_i] = mean_pos
            variances_neg[t_i] = var_neg
            means_neg[t_i] = mean_neg
            percentage[t_i] = len(states_action_pos) / len(states_pos)
            states_stats.append(
                {"dists_pos": dist, "dir_pos": direction,
                 "dists_neg": dist_neg, "dir_ab_neg": direction_neg,
                 "means": mean_pos, "variances": var_pos,
                 "position_pos": data_A.squeeze(), "position_neg": data_A_neg.squeeze(),
                 "action_type": action_type, "mask_type": mask_type, "prop_type": prop_type, "obj_types": obj_type,
                 "indices": mask_action_state_pos})

        variances_ranked, v_rank = variances.sort()
        # state_num = states.shape[0]
        # passed_indices = torch.zeros(state_num, dtype=torch.bool)
        # v_i = 0
        # coverage = 0
        # skill_n_stats = []
        # percentage_each_stat = []
        # while coverage < 0.99 or v_i < len(variances_ranked):
        #     if variances_ranked<
        #     skill_indices = states_stats[v_rank[v_i]]["indices"]
        #
        #     stat_cover_state_indices = torch.tensor(rewards_indices_pos)[skill_indices].unique()
        #
        #     new_states_percent = (~passed_indices[stat_cover_state_indices]).sum() / state_num
        #     percentage_each_stat.append(new_states_percent.tolist())
        #     passed_indices[stat_cover_state_indices] = True
        #     coverage = passed_indices.sum() / state_num
        #     skill_n_stats.append(states_stats[v_rank[v_i]])
        #     v_i += 1
        # passed_stats += skill_n_stats

        passed_variances = variances_ranked < var_th
        passed_comb_indices = v_rank[passed_variances]
        skill_n_stats = [states_stats[s_i] for s_i in passed_comb_indices]
        skill_n_passed_combs = [type_combs[s_i] for s_i in passed_comb_indices]
        passed_stats += skill_n_stats
    behs = []
    for state_stat in passed_stats:
        indices = state_stat["indices"]
        dist_pos = state_stat["dists_pos"]
        dir_value = state_stat["dir_pos"]
        action_ids = state_stat["action_type"].tolist()

        dist_range = [math_utils.get_90_percent_range_2d(pos).tolist() for pos in dist_pos.numpy()]
        if np.abs(dist_range).max() > max_dist:
            continue
        dist_one_percent_value = math_utils.closest_one_percent(dist_pos)
        dir_quarter_value = math_utils.closest_quarter(dir_value)
        x_types = []
        y_types = []
        x_confs = []
        y_confs = []
        direction_types = []
        direction_confs = []
        for frame_i in range(len(dist_one_percent_value)):
            x_one_percent_types, x_counts = dist_one_percent_value[frame_i, :, 0].unique(return_counts=True)
            y_one_percent_types, y_counts = dist_one_percent_value[frame_i, :, 1].unique(return_counts=True)
            x_types.append(x_one_percent_types.tolist())
            y_types.append(y_one_percent_types.tolist())
            x_confs.append((x_one_percent_types / x_one_percent_types.sum()).tolist())
            y_confs.append((y_one_percent_types / y_one_percent_types.sum()).tolist())

            dir_quarter_values, dir_counts = dir_quarter_value[frame_i].unique(return_counts=True)
            direction_types.append(dir_quarter_values.tolist())
            direction_confs.append((dir_counts / dir_counts.sum()).tolist())

        behs.append({
            "x_range": x_types,
            "y_range": y_types,
            "x_conf": x_confs,
            "y_conf": y_confs,
            "dir_range": direction_types,
            "dir_conf": direction_confs,
            "skill_len": len(x_types),
            "dists_pos": state_stat["dists_pos"].tolist(),
            "dir_pos": state_stat["dir_pos"].tolist(),
            "position_pos": state_stat["position_pos"].tolist(),
            "position_neg": state_stat["position_neg"].tolist(),
            "dists_neg": state_stat["dists_neg"].tolist(),
            "dir_ab_neg": state_stat["dir_ab_neg"].tolist(),
            "means": state_stat["means"].tolist(),
            "variance": state_stat["variances"].tolist(),
            "action_type": state_stat["action_type"].tolist(),
            "masks": state_stat["mask_type"].tolist(),
            "obj_combs": state_stat["obj_types"],
            "prop_combs": state_stat["prop_type"],
            "rewards": None,
        })

    return behs


def stat_negative_rewards(states, actions, rewards, zero_reward, game_info, prop_indices, var_th):
    mask_neg_reward = rewards < zero_reward
    neg_rewards = rewards[mask_neg_reward]

    neg_states = states[mask_neg_reward]
    neg_actions = actions[mask_neg_reward]
    neg_masks = mask_tensors_from_states(neg_states, game_info)

    pos_states = states[~mask_neg_reward]
    pos_actions = actions[~mask_neg_reward]
    pos_masks = mask_tensors_from_states(pos_states, game_info)

    neg_mask_types = neg_masks.unique(dim=0)
    neg_action_types = neg_actions.unique()
    obj_types = get_all_2_combinations(game_info, reverse=False)
    prop_types = [prop_indices]
    type_combs = list(itertools.product(neg_mask_types, neg_action_types, obj_types, prop_types))

    states_stats = []
    variances = []
    means = []
    percentage = torch.zeros(len(type_combs))
    for t_i in range(len(type_combs)):
        mask_type, action_type, obj_type, prop_type = type_combs[t_i]

        # calc distance between any two objects
        # mask_type_repeated = torch.repeat_interleave(mask_type.unsqueeze(0), len(neg_masks), 0)
        action_state_indices = ((neg_actions == action_type) * (neg_masks == mask_type).prod(-1).bool())
        action_pos_indices = ((pos_actions == action_type) * (pos_masks == mask_type).prod(-1).bool())

        action_states = neg_states[action_state_indices]
        action_states_pos = pos_states[action_pos_indices]

        obj_0_indices = game_info[obj_type[0]]["indices"]
        obj_1_indices = game_info[obj_type[1]]["indices"]
        obj_combs = torch.tensor(list(itertools.product(obj_0_indices, obj_1_indices)))
        obj_a_indices = obj_combs[:, 0].unique()
        obj_b_indices = obj_combs[:, 1].unique()
        if len(obj_a_indices) == 1:
            data_A = action_states[:, obj_a_indices][:, :, prop_type]
            data_B = action_states[:, obj_b_indices][:, :, prop_type]
            dist, b_index = math_utils.dist_a_and_b_closest(data_A, data_B)
            dir_ab = math_utils.dir_a_and_b_with_alignment(data_A, data_B, b_index)
            dist_dir_pos = torch.cat((dist, dir_ab), dim=1)
            data_A_pos = action_states_pos[:, obj_a_indices][:, :, prop_type]
            data_B_pos = action_states_pos[:, obj_b_indices][:, :, prop_type]
            dist_pos, b_pos_index = math_utils.dist_a_and_b_closest(data_A_pos, data_B_pos)
            dir_pos_ab = math_utils.dir_a_and_b_with_alignment(data_A_pos, data_B_pos, b_pos_index)
            dist_dir_neg = torch.cat((dist_pos, dir_pos_ab), dim=1)
        else:
            dist = torch.zeros(2)
            dist_pos = torch.zeros(2)
            dir_ab = torch.zeros(2)
            dir_pos_ab = torch.zeros(2)
        var, mean = torch.var_mean(dist_dir_pos)
        var_pos, mean_pos = torch.var_mean(dist_dir_neg)
        if len(dist) < 3 or var_pos == 0 or dist.sum() == 0:
            var = 1e+20
            mean = 1e+20
        variances.append(var)
        means.append(mean)
        percentage[t_i] = len(action_states) / len(neg_states)
        states_stats.append(
            {"dists": dist, "dir": dir_ab, "dists_pos": dist_pos, "dir_ab_pos": dir_pos_ab, "means": mean,
             "variances": var, "action_type": action_type,
             "mask_type": mask_type, "prop_type": prop_type, "obj_types": obj_type,
             "indices": action_state_indices})
    variances_ranked, v_rank = torch.tensor(variances).sort()

    passed_variances = variances_ranked < var_th
    passed_comb_indices = v_rank[passed_variances]
    passed_stats = [states_stats[s_i] for s_i in passed_comb_indices]

    neg_behs = []
    for state_stat in passed_stats:
        indices = state_stat["indices"]
        neg_behs.append({
            "dists": state_stat["dists"].tolist(),
            "dir": state_stat["dir"].tolist(),
            "dists_pos": state_stat["dists_pos"].tolist(),
            "dir_ab_pos": state_stat["dir_ab_pos"].tolist(),
            "obj_combs": state_stat["obj_types"],
            "prop_combs": state_stat["prop_type"],
            "means": state_stat["means"].tolist(),
            "variance": state_stat["variances"].tolist(),
            "rewards": neg_rewards[indices].tolist(),
            "action_type": state_stat["action_type"].tolist(),
            "masks": state_stat["mask_type"].tolist()
        })

    return neg_behs


def top_k_percent(scores, top_kp):
    score_sum = 0
    indices = []
    for s_i, score in enumerate(scores):
        score_sum += score
        indices.append(s_i)
        if score_sum > top_kp:
            break

    return indices


def brute_search(args, facts, fact_truth_table, actions, rewards):
    learned_behaviors = []
    for at_i, action_type in enumerate(actions.unique()):
        fact_table = fact_truth_table[actions == at_i, :]
        fact_anti_table = fact_truth_table[actions != at_i, :]
        action_rewards = rewards[actions == at_i]
        learned_action_behivors = []
        for fact_len in range(1):
            fact_combs = get_fact_combs(fact_len + 1, fact_table.size(1))
            fact_pos_num, fact_neg_num, pos_states, neg_states, fact_rewards = stat_facts_in_states(
                fact_combs, fact_len, fact_table, fact_anti_table, action_rewards)
            fact_comb_score = fact_pos_num / (fact_neg_num + fact_pos_num + 1e-20)
            scores, score_rank = fact_comb_score.sort(descending=True)
            rank_score_indices = scores > args.fact_conf
            if args.with_explain:
                print(f"(Reasoning Path Finding Behavior) act: {action_type}, "
                      f"facts: {fact_len + 1}, "
                      f"states: {fact_pos_num[score_rank][rank_score_indices]}, "
                      f"scores: {scores[rank_score_indices]}")
            pos_fact_indices = score_rank[rank_score_indices]
            for index in pos_fact_indices:
                for fact_index in fact_combs[index]:
                    print(f"action {action_type} comb {index}, {facts[fact_index]}, score {fact_comb_score[index]:.2f}")
            pos_state_num = fact_pos_num[pos_fact_indices]
            pos_state_total = [len(fact_table)] * len(pos_state_num)
            neg_state_num = fact_neg_num[pos_fact_indices]
            neg_state_total = [len(fact_anti_table)] * len(neg_state_num)
            pos_rewards = fact_rewards[pos_fact_indices]

            for p_i in range(len(pos_fact_indices)):
                pos_facts = [facts[i] for i in fact_combs[pos_fact_indices][p_i].reshape(-1)]

                mask_state_pos = torch.zeros(len(fact_truth_table), dtype=torch.bool)
                mask_state_pos[actions == at_i] = pos_states[pos_fact_indices[p_i]]

                mask_state_neg = torch.zeros(len(fact_truth_table), dtype=torch.bool)
                mask_state_neg[actions != at_i] = neg_states[pos_fact_indices[p_i]]

                learned_action_behivors.append({"facts": pos_facts,
                                                "mask_state_pos": mask_state_pos.tolist(),
                                                "mask_state_neg": mask_state_neg.tolist(),
                                                "score": scores[rank_score_indices][p_i].tolist(),
                                                "expected_reward": pos_rewards[p_i].tolist(),
                                                "passed_state_num": pos_state_num[p_i].tolist(),
                                                "test_passed_state_num": pos_state_total[p_i],
                                                "failed_state_num": neg_state_num[p_i].tolist(),
                                                "test_failed_state_num": neg_state_total[p_i]})

        learned_behaviors.append(learned_action_behivors)
    return learned_behaviors


def stat_pos_data(args, states, actions, rewards, game_info, prop_indices):
    facts = get_all_facts(game_info, prop_indices)
    truth_table_file = args.check_point_path / f"{args.m}_truth_table.pt"
    if os.path.exists(truth_table_file):
        truth_table = torch.load(truth_table_file)
    else:
        truth_table = check_fact_truth(facts, states, actions, game_info)
        torch.save(truth_table, truth_table_file)

    behavior_data = brute_search(args, facts, truth_table, actions, rewards)

    return behavior_data


def best_pos_data_comb(pos_beh_data):
    behavior_data = []

    # prune data based on top score if they have same mask
    for a_i, action_data in enumerate(pos_beh_data):
        masks = torch.tensor([fact['mask'] for data in action_data for fact in data['facts']])
        mask_types = masks.unique(dim=0)
        for mask_type in mask_types:
            mask_type_indices = (masks == mask_type).prod(dim=1).bool()
            data_mask = [action_data[d_i] for d_i in range(len(action_data)) if mask_type_indices[d_i]]
            data_best_index = torch.tensor([data['score'] for data in data_mask]).argmax()
            data_best = data_mask[data_best_index]
            data_best['action'] = a_i
            behavior_data.append(data_best)

    return behavior_data


def extract_fact_data(fact, frame_state):
    obj_a, obj_b = fact.obj_comb
    props = fact.prop_comb
    assert len(fact.preds) == 1
    dist = torch.abs(frame_state[0, obj_a, props] - frame_state[0, obj_b, props]).reshape(1, -1)
    return dist
