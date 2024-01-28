# Created by jing at 23.12.23
import torch
from tqdm import tqdm
from src import config
from pi import predicate
from pi import pi_lang
from itertools import combinations


def get_param_range(min, max, unit):
    length = (max - min) // unit
    if length == 0:
        return torch.zeros(1)

    space = torch.zeros(int(length))
    for v_i in range(len(space)):
        space[v_i] = min + unit * v_i
    return space


def get_all_2_combinations(game_info):
    all_combinations = [[i_1, i_2] for i_1, s_1 in enumerate(game_info) for i_2, s_2 in
                        enumerate(game_info) if s_1 != s_2]
    return all_combinations


def get_all_facts(game_info, prop_indices):
    relate_2_obj_types = get_all_2_combinations(game_info)
    relate_2_prop_types = [[each] for each in prop_indices]
    facts = get_smp_facts(game_info, relate_2_obj_types, relate_2_prop_types)
    print(f"fact number: {len(facts)}")
    return facts


def get_fact_obj_combs(game_info, fact):
    type_0_index = fact["objs"][0]
    type_1_index = fact["objs"][1]
    _, obj_0_indices, _ = game_info[type_0_index]
    _, obj_1_indices, _ = game_info[type_1_index]
    obj_combs = enumerate_two_combs(obj_0_indices, obj_1_indices)
    return obj_combs


def get_fact_mask(fact, repeat):
    fact_mask_tensor = mask_name_to_tensor(fact["mask"], config.mask_splitter)
    fact_mask_tensor = torch.repeat_interleave(fact_mask_tensor.unsqueeze(0), repeat, 0)
    return fact_mask_tensor


def fact_is_true(states, fact, game_info):
    prop = fact["props"]
    obj_combs = get_fact_obj_combs(game_info, fact)
    fact_mask = get_fact_mask(fact, len(states))
    state_mask = mask_tensors_from_states(states, game_info)
    fact_satisfaction = (fact_mask == state_mask).prod(dim=-1).bool()

    # fact_satisfaction = torch.zeros(len(states), dtype=torch.bool)
    pred_satisfaction = torch.zeros(len(states), dtype=torch.bool)
    for obj_a, obj_b in obj_combs:
        data_A = states[:, obj_a, prop].reshape(-1)
        data_B = states[:, obj_b, prop].reshape(-1)

        state_pred_satisfaction = torch.zeros((3, len(states)), dtype=torch.bool)
        for i in range(len(fact['preds'])):
            state_pred_satisfaction[i] = fact['preds'][i].eval(data_A, data_B)

        pred_satisfaction += state_pred_satisfaction.prod(dim=0).bool()

    fact_satisfaction *= pred_satisfaction

    # satisfaction = mask_satisfaction * fact_satisfaction
    assert len(fact_satisfaction) == len(states)
    return fact_satisfaction


def check_fact_truth(facts, data, game_info):
    truth_table = torch.zeros((len(data), len(facts)), dtype=torch.bool)

    states = torch.tensor([d[0].tolist() for d in data]).squeeze()
    for f_i in tqdm(range(len(facts)), ascii=True, desc="Fact Table"):
        fact_satisfaction = fact_is_true(states, facts[f_i], game_info)
        truth_table[:, f_i] = fact_satisfaction

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


def comb_buffers(states, actions, rewards, reason_source):
    assert len(states) == len(actions) == len(rewards) == len(reason_source)
    data = []
    for i in range(len(states)):
        data.append([states[i].unsqueeze(0), actions[i], rewards[i], reason_source[i]])
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


def mask_tensors_from_states(states, game_info):
    mask_tensors = torch.zeros((len(states), len(game_info)), dtype=torch.bool)
    for i in range(len(game_info)):
        name, obj_indices, prop_index = game_info[i]
        obj_exist_counter = states[:, obj_indices, prop_index].sum()
        mask_tensors[:, i] = obj_exist_counter > 0
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
        for al_i in range(len(pred_lists[1])):
            for am_i in range(len(pred_lists[2])):
                pred_combs.append([pred_lists[0][dir_i], pred_lists[1][al_i], pred_lists[2][am_i]])
    return pred_combs


def get_smp_facts(game_info, relate_2_obj_types, relate_2_prop_types):
    at_least_preds = predicate.get_at_least_preds(1, config.dist_num, config.max_dist)
    at_most_preds = predicate.get_at_most_preds(1, config.dist_num, config.max_dist)
    dir_preds = [predicate.GT()]
    pred_combs = all_pred_combs([dir_preds, at_least_preds, at_most_preds])
    facts = []
    obj_names = [name for name, _, _ in game_info]
    for type_comb in relate_2_obj_types:
        masks = exist_mask_names(obj_names, type_comb)
        for mask in masks:
            for prop_types in relate_2_prop_types:
                for p_i in range(len(pred_combs)):
                    fact_preds = pred_combs[p_i]
                    facts.append({"mask": mask, "objs": type_comb, "props": prop_types,
                                  "pred_tensors": torch.ones((3), dtype=torch.bool), "preds": fact_preds})

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
    print(f"fact combinations: {len(comb)}")
    return comb


def fact_grouping_by_head(facts):
    head_id = -1
    checked_head = []
    fact_head_ids = []
    fact_bodies = []
    fact_heads=[]
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
