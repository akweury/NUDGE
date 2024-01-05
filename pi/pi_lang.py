# Created by shaji at 30/11/2023


from nsfr.nsfr.fol.logic import InvPredicate
from nsfr.nsfr.fol.language import DataType
from nsfr.nsfr.fol.logic import Atom, Clause, Var

from pi import predicate
from src import config


def extract_behavior_terms(args, behavior):
    terms = []
    for obj_code in behavior["grounded_types"]:
        obj_name = args.state_names[obj_code]
        terms.append(Var(obj_name))

    return terms


def generate_action_predicate(args, behavior):
    if "counter_action" in behavior.keys():
        action_code = behavior["counter_action"].argmax()
        action_name = args.counter_action_names[action_code]
        action_predicate = InvPredicate(action_name, 1, [DataType("agent")], config.counter_action_pred_name)
    elif "action" in behavior.keys():
        action_code = behavior["action"]
        action_name = args.action_names[action_code]
        action_predicate = InvPredicate(action_name, 1, [DataType("agent")], config.action_pred_name)
    else:
        raise ValueError

    return action_predicate


def generate_exist_predicate(existence, obj_name):
    pred_name = existence
    dtypes = [DataType(dt) for dt in [obj_name]]
    pred = InvPredicate(pred_name, 1, dtypes, config.exist_pred_name)
    return pred


def generate_func_predicate(args, behavior, p_i):
    obj_A = args.state_names[behavior["grounded_types"][0]]
    obj_B = args.state_names[behavior["grounded_types"][1]]
    prop_name = args.prop_names[behavior['grounded_prop'][0]]
    pred = behavior["preds"][p_i]
    pred_func_name = pred.name

    pred_name = obj_A + "_" + prop_name + "_" + pred_func_name + "_" + obj_B

    dtypes = [DataType(dt) for dt in [obj_A, obj_B]]
    pred = InvPredicate(pred_name, 2, dtypes, config.func_pred_name,
                        grounded_prop=prop_name,
                        grounded_objs=[obj_A, obj_B],
                        pred_func=pred_func_name,
                        parameter_min=pred.p_bound['min'],
                        parameter_max=pred.p_bound['max'])
    return pred


def behavior_action_as_head_atom(args, behavior):
    action_predicate = generate_action_predicate(args, behavior)
    head_atom = Atom(action_predicate, [Var("agent")])
    return head_atom


def behavior_predicate_as_func_atom(args, behavior):
    # behavior['grounded_objs'] determines terms in the clause
    terms = extract_behavior_terms(args, behavior)
    func_atom = []
    for p_i, pred in enumerate(behavior['preds']):
        if behavior["p_satisfication"][p_i]:
            func_pred = generate_func_predicate(args, behavior, p_i)
            func_atom.append(Atom(func_pred, terms))
    return func_atom


def behavior_existence_as_env_atoms(args, behavior):
    obj_existence = behavior["mask"].split(config.mask_splitter)
    exist_atoms = []
    for exist_obj in obj_existence:

        if "not_exist" in exist_obj:
            existence = "not_exist"
            obj_name = exist_obj.split("not_exist_")[1]
        else:
            existence = "exist"
            obj_name = exist_obj.split("exist_")[1]
        pred = generate_exist_predicate(existence, obj_name)
        exist_atom = Atom(pred, [Var(obj_name)])
        exist_atoms.append(exist_atom)

    return exist_atoms


def behavior2clause(args, behavior):
    # behavior['action'] determines head atom in the clause
    head_atom = behavior_action_as_head_atom(args, behavior)
    # behavior['grounded_prop'] and strategy['pred'] determine the predicate as functional atom in the clause
    func_atom = behavior_predicate_as_func_atom(args, behavior)
    # behavior['mask'] determine the object existence as environment atoms in the clause
    env_atoms = behavior_existence_as_env_atoms(args, behavior)

    body_atoms = func_atom + env_atoms
    new_clause = Clause(head_atom, body_atoms)

    return new_clause


def behaviors2clauses(args, behaviors):
    clauses = []
    for behavior in behaviors:
        if behavior["is_grounded"]:
            clause = behavior2clause(args, behavior)
            clauses.append(clause)
        # clause_weights.append()
    print('======= Clauses from Behaviors ======')
    for c in clauses:
        print(c)
    return clauses
