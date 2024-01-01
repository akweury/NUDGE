# Created by jing at 27.11.23

import torch
from itertools import compress

from pi import pi_lang, predicate
from pi.game_settings import get_idx, get_state_names


def split_data_by_action(states, actions):
    action_types = torch.unique(actions)
    data = {}
    for action_type in action_types:
        index_action = actions == action_type
        states_one_action = states[index_action]
        data[action_type] = states_one_action.squeeze()
    return data


def get_all_subsets(input_list):
    result = [[]]

    for elem in input_list:
        new_subsets = [subset + [elem] for subset in result]
        result.extend(new_subsets)

    return result


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


def all_exist_mask(states, state_names):
    primitive_masks = {}
    for s_1i, s_1name in enumerate(state_names):
        mask1_exist = states[:, s_1i, s_1i] > 0
        # mask1_not_exist = states[:, s_1i, s_1i] == 0
        primitive_masks[f'{s_1name}'] = mask1_exist
        # primitive_masks[f'{s_1name}_not_exist'] = mask1_not_exist

    masks = {}
    switches = get_all_subsets(state_names)
    for switch in switches:
        switch_masks = []
        for name in state_names:
            mask = ~primitive_masks[name]
            if name in switch:
                mask = ~mask
            switch_masks.append(mask.tolist())
        switch_mask = torch.prod(torch.tensor(switch_masks).float(), dim=0).bool()
        mask_name = gen_mask_name(switch, state_names)
        masks[mask_name] = switch_mask
        print(f'{mask_name}: {torch.sum(switch_mask)}')

    return masks


def state_exist_mask(state, state_names):
    mask = {}
    for s_1i, s_1name in enumerate(state_names):
        mask1_exist = state[:, s_1i, s_1i] > 0
        mask[f'{s_1name}'] = mask1_exist
    return mask


def micro_program2behaviors(args, data):
    # micro-programs
    # TODO: print micro-programs structure
    prop_list = get_idx(args)
    state_name_list = get_state_names(args)

    state_relate_2_aries = [[i_1, i_2] for i_1, s_1 in enumerate(state_name_list) for i_2, s_2 in
                            enumerate(state_name_list) if s_1 != s_2]

    behaviors = []
    for action, states in data.items():
        masks = all_exist_mask(states, state_name_list)
        for objs in state_relate_2_aries:
            for mask_name, mask in masks.items():
                for prop_idx in prop_list:
                    # select data
                    data_A = states[mask, objs[0]]
                    data_B = states[mask, objs[1]]
                    if len(data_A) == 0:
                        continue
                    data_A = data_A[:, prop_idx]
                    data_B = data_B[:, prop_idx]
                    # distinguish predicates
                    all_preds = predicate.get_preds()
                    p_satisfication = torch.zeros(len(all_preds), dtype=torch.bool)
                    for p_i, pred in enumerate(all_preds):
                        p_satisfication[p_i] = pred.fit(data_A, data_B, objs)
                    if (p_satisfication.sum()) > 0:
                        print(f'new pred, grounded_objs:{objs}, action:{action}')
                        behavior = {'preds': all_preds, 'p_satisfication': p_satisfication,
                                    'grounded_objs': objs, 'grounded_prop': prop_idx,
                                    'action': action, 'mask': mask_name}
                        behaviors.append(behavior)

    return behaviors


def split_counter_actions(counter_action):
    counter_action_types = torch.unique(counter_action, dim=0)
    counter_action_dict = {}
    for counter_action_type in counter_action_types:
        indices = ((counter_action == counter_action_type).sum(dim=1)) == counter_action.size(1)
        counter_action_dict[counter_action_type] = indices

    return counter_action_dict


def find_pred_parameters_in_smps(smps, pred):
    parameter = None
    for smp in smps:
        if smp.pred_funcs[0] == pred:
            parameter = smp.p_bound
    return parameter


def split_counter_behaviors(counter_behaviors):
    counter_actions = []
    neural_actions = []
    splitted_behaviors = []

    # split based on smps
    for behavior in counter_behaviors:
        counter_actions.append(behavior.counter_action.reshape(1, -1))
        neural_actions.append(behavior.neural_action.reshape(1, -1))
    counter_actions = torch.cat(counter_actions, dim=0)
    neural_actions = torch.cat(neural_actions, dim=0)

    counter_action_types = torch.unique(counter_actions, dim=0)
    neural_action_types = torch.unique(neural_actions, dim=0)

    for counter_action_type in counter_action_types:
        # get indices of each type of counter actions
        counter_action_indices = ((counter_actions == counter_action_type).sum(dim=1)) == counter_actions.size(1)
        for neural_action_type in neural_action_types:
            # get indices of each type of neural actions
            neural_action_indices = ((neural_actions == neural_action_type).sum(dim=1)) == neural_actions.size(1)
            indices = counter_action_indices * neural_action_indices
            same_type_behaviors = list(compress(counter_behaviors, indices))
            if len(same_type_behaviors) == 0:
                continue
            splitted_behaviors.append(same_type_behaviors)

    return splitted_behaviors


def common_prop_behaviors(behaviors, prop_name):
    value_space = [eval(f"explain.{prop_name}") for explain in behaviors[0].explains]
    common_prop = []
    common_prop_behaviors = []
    # get types of counter actions
    for value in value_space:
        is_common_prop = True
        common_prop_state_behavior = []
        for behavior in behaviors:
            exist_common_prop = False
            for explain in behavior.explains:
                if eval(f"explain.{prop_name}") == value:
                    exist_common_prop = True
                    common_prop_state_behavior.append(explain)
                    break

            if not exist_common_prop:
                is_common_prop = False
                break

        if is_common_prop:
            common_prop.append(value)
            common_prop_behaviors.append(common_prop_state_behavior)

    return common_prop_behaviors, common_prop


# def micro_program2counteract_params(args, neural_actions, states, behavior_smps):
#     # actions based on the neural agent
#     neural_actions = neural_actions.squeeze()
#     neural_actions[neural_actions > 0.8] = 1
#     neural_actions[neural_actions < 0.8] = 0
#
#     behavior_actions, params = pred_action_by_smps(args, behavior_smps, states)
#     # if more than 1 action is possible for the current state
#
#     multiple_action_states = behavior_actions.sum(dim=1).sum(dim=1) > 1
#     smp_counteract_params = extract_smp_counteract_params(args, behavior_smps,
#                                                         states[multiple_action_states],
#                                                         behavior_actions[multiple_action_states],
#                                                         neural_actions[multiple_action_states],
#                                                         params[multiple_action_states])
#     # counteract_behavior = prune_behaviors(counteract_behavior)
#     return smp_counteract_params


def buffer2behaviors(args, buffer):
    # data preparation
    actions = buffer.actions
    states = buffer.logic_states
    data = split_data_by_action(states, actions)
    behaviors = micro_program2behaviors(args, data)

    return behaviors


def buffer2clauses(args, buffer):
    agent_behaviors = buffer2behaviors(args, buffer)
    clauses = pi_lang.behaviors2clauses(args, agent_behaviors)
    return clauses
