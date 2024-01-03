# Created by jing at 01.12.23

import torch

import pi.game_env
from pi import behavior, predicate
from pi.MicroProgram import MicroProgram
from pi.utils import args_utils, smp_utils, log_utils

from src import config


def extract_action(args, behavior):
    action = torch.zeros(len(args.action_names))
    action[behavior['action']] = 1
    return action


def extract_existence_mask(args, behavior):
    mask = torch.ones(len(args.state_names), dtype=torch.bool)
    existences = behavior['mask'].split(config.mask_splitter)
    for e_i, existence in enumerate(existences):
        if 'not' in existence:
            mask[e_i] = False
    return mask


def extract_pred_func_data(args, behavior):
    pred_funcs = behavior['pred']
    prop_codes = []
    obj_codes = behavior['grounded_objs']
    parameters = behavior['p_bound']
    for atom in behavior.body:
        if atom.pred.p_type == config.func_pred_name:
            grounded_obj = atom.pred.grounded_objs
            grounded_prop = atom.pred.grounded_prop
            pred_func_name = atom.pred.pred_func
            parameters = atom.pred.p_bound

            grounded_prop_code = args.prop_names.index(grounded_prop)
            grounded_obj_codes = [args.state_names.index(obj) for obj in grounded_obj]
            pred_func = predicate.pred_dict[pred_func_name]

            pred_funcs.append(pred_func)
            prop_codes.append(grounded_prop_code)
            obj_codes.append(grounded_obj_codes)

    return obj_codes, prop_codes, pred_funcs, parameters


def behavior2smp(args, behavior):
    action = extract_action(args, behavior)
    existence_mask = extract_existence_mask(args, behavior)
    p_spaces = []
    for p_i, pred in enumerate(behavior["preds"]):
        p_spaces.append(smp_utils.get_param_range(pred.p_bound['min'], pred.p_bound['max'], config.smp_param_unit))
    smp = MicroProgram(args.obj_type_names,
                       action,
                       existence_mask,
                       behavior['grounded_types'],
                       behavior['grounded_prop'],
                       behavior["preds"],
                       p_spaces,
                       behavior["p_satisfication"])
    return smp


def behavior2smps(args, buffer, behaviors):
    smps = []
    for behavior in behaviors:
        smp = behavior2smp(args, behavior)
        smps.append(smp)

    # update parameters in smps
    rectify_smps(args, buffer, smps)

    return smps


def extract_smp_counteract_params(smps, behavior_actions, neural_actions, counter_params):
    # smp_counter_params = {i: [] for i in range(len(smps))}
    for state_i in range(len(behavior_actions)):
        counter_actions = behavior_actions[state_i]
        neural_action = neural_actions[state_i]
        state_params = counter_params[state_i]

        for smp_i, smp in enumerate(smps):
            counter_action = counter_actions[smp_i]
            # if counter action is not equal to neural action
            if counter_action.sum() > 0 and torch.prod(counter_action == neural_action) == 0:
                # counter action should not be inferred under such scenarios
                smp_params = state_params[smp_i]
                for p_i, p_space in enumerate(smp.p_spaces):
                    p_param = smp_params[p_i]
                    if p_param in p_space:
                        smp.p_spaces[p_i] = torch.where(p_space == p_param, 0, p_space)


def pred_action_by_smps(args, smps, states):
    behavior_actions = torch.zeros(size=(len(smps), states.size(0), len(args.action_names))).to(args.device)
    p_parameters = []
    for s_i, smp in enumerate(smps):
        action, p_params = smp(states, config.obj_type_indices_getout)
        behavior_actions[s_i] = action
        p_params = torch.cat(p_params, dim=0)
        p_parameters.append(p_params.unsqueeze(0))
    p_parameters = torch.cat(p_parameters, dim=0)
    parameters = p_parameters.permute(2, 0, 1)
    behavior_actions = behavior_actions.permute(1, 0, 2)

    return behavior_actions, parameters


def rectify_smp(smp, params):
    if len(params) == 0:
        return
    params = torch.round(params, decimals=2)
    for param in params:
        if param in smp.p_space:
            smp.p_space = torch.where(smp.p_space == param, 0, smp.p_space)


def rectify_smps(args, buffer, behavior_smps):
    neural_actions = buffer.action_probs.squeeze()
    neural_actions[neural_actions > 0.8] = 1
    neural_actions[neural_actions < 0.8] = 0
    states = buffer.logic_states

    # predict actions based on smps
    behavior_actions, params = pred_action_by_smps(args, behavior_smps, states)

    # indices of counteract states
    behavior_types = behavior_actions.sum(dim=1) / (behavior_actions.sum(dim=1) + 1e-20)
    cs_indices = behavior_types.sum(dim=1) > 1
    log_utils.add_lines(f'(before rectification) counteract state number: {cs_indices.sum()}', args.log_file)

    # the correspondence counteract params
    extract_smp_counteract_params(behavior_smps, behavior_actions[cs_indices], neural_actions[cs_indices],
                                  params[cs_indices])

    # indices of counteract states
    behavior_actions, params = pred_action_by_smps(args, behavior_smps, states)
    behavior_types = behavior_actions.sum(dim=1) / (behavior_actions.sum(dim=1) + 1e-20)
    cs_indices = behavior_types.sum(dim=1) > 1
    log_utils.add_lines(f'(after rectification) counteract state number: {cs_indices.sum()}', args.log_file)
    # smp_counteract_params = micro_program2counteract_params(args, actions, states, behavior_smps)
    # using counteract behaviors to rectify smp parameters
    # for s_i, smp in enumerate(behavior_smps):
    #     params = smp_counteract_params[s_i]
    #     rectify_smp(smp, params)
