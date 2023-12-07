# Created by jing at 01.12.23

import torch
from pi import micro_program_search, predicate
from pi.MicroProgram import MicroProgram
from pi.utils import args_utils

from src import config

def extract_action_code(args, clause):
    action_name = clause.head.pred.name
    action_code = args.action_names.index(action_name)
    return torch.tensor(action_code)


def extract_existence_mask(args, clause):
    mask = torch.zeros(len(args.state_names), dtype=torch.bool)
    for atom in clause.body:
        if atom.pred.p_type == config.exist_pred_name:
            if atom.pred.name == "exist":
                obj_name = atom.pred.dtypes[0].name
                obj_id = args.state_names.index(obj_name)
                mask[obj_id] = True

    return mask


def extract_pred_func_data(args, clause):
    pred_funcs = []
    prop_codes = []
    obj_codes = []

    for atom in clause.body:
        if atom.pred.p_type == config.func_pred_name:
            grounded_obj = atom.pred.grounded_objs
            grounded_prop = atom.pred.grounded_prop
            pred_func_name = atom.pred.pred_func

            grounded_prop_code = args.prop_names.index(grounded_prop)
            grounded_obj_codes = [args.state_names.index(obj) for obj in grounded_obj]
            pred_func = predicate.pred_dict[pred_func_name]

            pred_funcs.append(pred_func)
            prop_codes.append(grounded_prop_code)
            obj_codes.append(grounded_obj_codes)

    return obj_codes, prop_codes, pred_funcs


def build_smp(action, mask, obj_codes, prop_codes, pred_funcs):
    smp = MicroProgram(action, mask, obj_codes, prop_codes, pred_funcs)
    return smp


def clause2smp(args, clause):
    action = extract_action_code(args, clause)
    existence_mask = extract_existence_mask(args, clause)
    obj_codes, prop_codes, pred_funcs = extract_pred_func_data(args, clause)
    smp = build_smp(action, existence_mask, obj_codes, prop_codes, pred_funcs)
    return smp


def clauses2smps(args, clauses):
    smps = []
    for clause in clauses:
        smp = clause2smp(args, clause)
        smps.append(smp)
    return smps


if __name__ == "__main__":
    args = args_utils.load_args(config.path_exps)
    buffer = micro_program_search.load_buffer(args)
    clauses = micro_program_search.buffer2clauses(args, buffer)

    smps = clauses2smps(args, clauses)
    print("program finished!")
