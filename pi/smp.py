# Created by jing at 01.12.23

import torch
from pi import behavior, predicate
from pi.MicroProgram import MicroProgram
from pi.utils import args_utils, smp_utils

from src import config

def extract_action(args, behavior):
    action = torch.zeros(len(args.action_names))
    action[behavior['action']] = 1
    return action


def extract_existence_mask(args, behavior):
    mask = torch.zeros(len(args.state_names), dtype=torch.bool)
    existences = behavior['mask'].split(config.mask_splitter)
    for e_i, existence in enumerate(existences):
        if 'exist' in existence:
            mask[e_i] = True
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
    smp = MicroProgram(action, existence_mask, behavior['grounded_objs'], behavior['grounded_prop'], behavior['pred'],
                       behavior['pred'].p_bound,
                       behavior['pred'].p_space)

    return smp


def behavior2smps(args, behaviors):
    smps = []
    for behavior in behaviors:
        smp = behavior2smp(args, behavior)
        smps.append(smp)
    return smps





def extract_smp_counteract_params(smps, behavior_actions, neural_actions, params):
    """ record a new behavior for each counter state """
    smp_counter_params = {i: [] for i in range(len(smps))}
    for state_i in range(len(behavior_actions)):
        counter_actions = behavior_actions[state_i]
        neural_action = neural_actions[state_i]
        param = params[state_i]

        for ca_i, counter_action in enumerate(counter_actions):
            if counter_action.sum() > 0 and torch.prod(counter_action == neural_action) == 0:
                smp_index = ca_i
                counter_param = param[ca_i].unsqueeze(0)
                smp_counter_params[smp_index].append(counter_param)
    for smp_i in smp_counter_params.keys():
        smp_counter_params[smp_i] = torch.cat(smp_counter_params[smp_i], dim=0)

    return smp_counter_params


def pred_action_by_smps(args, smps, states):
    behavior_actions = torch.zeros(size=(len(smps), states.size(0), len(args.action_names))).to(args.device)
    parameters = []
    for s_i, smp in enumerate(smps):
        action, params = smp(states, use_given_parameters=True)
        behavior_actions[s_i] = action
        parameters.append(params.unsqueeze(0))

    parameters = torch.cat(parameters, dim=0)
    parameters = parameters.permute(1, 0)
    behavior_actions = behavior_actions.permute(1, 0, 2)

    return behavior_actions, parameters


def rectify_smp(smp, params):
    smp_param_space = smp_utils.get_param_range(smp.p_bound['min'], smp.p_bound['max'], config.smp_param_unit)
    if smp_param_space is None:
        return
    params = torch.round(params, decimals=2)
    for param in params:
        if param in smp_param_space:
            smp_param_space = torch.where(smp_param_space == param, 0, smp_param_space)

    print('break')


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

    # the correspondence counteract params
    smp_counteract_params = extract_smp_counteract_params(behavior_smps, behavior_actions[cs_indices],
                                                          neural_actions[cs_indices], params[cs_indices])

    # smp_counteract_params = micro_program2counteract_params(args, actions, states, behavior_smps)
    # using counteract behaviors to rectify smp parameters
    for s_i, smp in enumerate(behavior_smps):
        params = smp_counteract_params[s_i]
        rectify_smp(smp, params)
    print('break')



if __name__ == "__main__":
    args = args_utils.load_args(config.path_exps)
    buffer = behavior.load_buffer(args)
    clauses = behavior.buffer2clauses(args, buffer)

    smps = behavior2smps(args, clauses)
    print("program finished!")