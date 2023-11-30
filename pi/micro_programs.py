# Created by jing at 27.11.23

import torch

from pi.game_settings import get_idx, get_state_names
from pi import predicate


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


def exist_mask(states, state_names):
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


def behaviors_from_micro_programs(args, data):

    # micro-programs
    # TODO: print micro-programs structure
    idx_list = get_idx(args)
    state_name_list = get_state_names(args)

    state_relate_2_aries = [[i_1, i_2] for i_1, s_1 in enumerate(state_name_list) for i_2, s_2 in
                            enumerate(state_name_list) if s_1 != s_2]

    behaviors = []
    for action, states in data.items():
        masks = exist_mask(states, state_name_list)
        for sr in state_relate_2_aries:
            for mask_name, mask in masks.items():
                for idx in idx_list:
                    # select data
                    data_A = states[mask, sr[0]]
                    data_B = states[mask, sr[1]]
                    if len(data_A) == 0:
                        continue
                    data_A = data_A[:, idx]
                    data_B = data_B[:, idx]
                    for pred in predicate.preds:
                        satisfy = pred(data_A, data_B, sr)
                        if satisfy:
                            print(f'new pred, grounded_objs:{sr}, action:{action}')
                            behavior = {'pred': pred,
                                        'grounded_objs': sr,
                                        'grounded_prop': idx,
                                        'action': action,
                                        'mask': mask_name
                                        }
                            behaviors.append(behavior)

    return behaviors


def buffer2behaviors(args, buffer):
    # data preparation
    actions = buffer.actions
    states = buffer.logic_states
    data = split_data_by_action(states, actions)
    behaviors = behaviors_from_micro_programs(args, data)

    return behaviors
