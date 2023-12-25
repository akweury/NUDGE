# Created by jing at 01.12.23

import torch
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
    p_space = smp_utils.get_param_range(behavior['pred'].p_bound['min'], behavior['pred'].p_bound['max'],
                                        config.smp_param_unit)
    smp = MicroProgram(action, existence_mask, behavior['grounded_objs'], behavior['grounded_prop'],
                       behavior['pred'], behavior['pred'].p_bound, p_space)

    return smp


def behavior2smps(args, buffer, behaviors):
    smps = []
    for behavior in behaviors:
        smp = behavior2smp(args, behavior)
        smps.append(smp)

    # update parameters in smps
    rectify_smps(args, buffer, smps)

    return smps


def extract_smp_counteract_params(smps, behavior_actions, neural_actions, params):
    # smp_counter_params = {i: [] for i in range(len(smps))}
    for state_i in range(len(behavior_actions)):
        counter_actions = behavior_actions[state_i]
        neural_action = neural_actions[state_i]
        param = params[state_i]

        for smp_i, counter_action in enumerate(counter_actions):
            # if counter action is not equal to neural action
            if counter_action.sum() > 0 and torch.prod(counter_action == neural_action) == 0:
                # counter action should not be inferred under such scenarios
                smp = smps[smp_i]
                parameter = param[smp_i]
                if parameter in smp.p_space:
                    smp.p_space = torch.where(smp.p_space == parameter, 0, smp.p_space)
                # smp_counter_params[smp_i].append(parameter.unsqueeze(0))
    # for smp_i in smp_counter_params.keys():
    #     if len(smp_counter_params[smp_i]) > 0:
    #         smp_counter_params[smp_i] = torch.cat(smp_counter_params[smp_i], dim=0)


def pred_action_by_smps(args, smps, states):
    behavior_actions = torch.zeros(size=(len(smps), states.size(0), len(args.action_names))).to(args.device)
    parameters = []
    for s_i, smp in enumerate(smps):
        action, params = smp(states)
        behavior_actions[s_i] = action
        parameters.append(params.unsqueeze(0))

    parameters = torch.cat(parameters, dim=0)
    parameters = parameters.permute(1, 0)
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


if __name__ == "__main__":
    args = args_utils.load_args(config.path_exps)
    buffer = behavior.load_buffer(args)
    clauses = behavior.buffer2clauses(args, buffer)

    smps = behavior2smps(args, clauses)
    print("program finished!")
