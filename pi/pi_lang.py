# Created by shaji at 30/11/2023


from nsfr.nsfr.fol.logic import InvPredicate
from nsfr.nsfr.fol.language import DataType
from nsfr.nsfr.fol.logic import Atom, Clause, Var

from pi import predicate
from src import config


def extract_fact_terms(args, fact):
    terms = []
    for obj_code in fact.obj_comb:
        obj_name, _, _ = args.obj_info[obj_code]
        terms.append(Var(obj_name))

    return terms


def generate_action_predicate(args, behavior):
    action_code = behavior.action
    action_name = args.action_names[action_code]
    if behavior.neg_beh:
        action_name = "not_" + action_name
    action_predicate = InvPredicate(action_name, 1, [DataType("agent")], config.action_pred_name)

    return action_predicate


def generate_exist_predicate(existence, obj_name):
    pred_name = existence
    dtypes = [DataType(dt) for dt in [obj_name]]
    pred = InvPredicate(pred_name, 1, dtypes, config.exist_pred_name)
    return pred


def generate_func_predicate(args, fact, pred_i):
    obj_A, _, _ = args.obj_info[fact.obj_comb[0]]
    obj_B, _, _ = args.obj_info[fact.obj_comb[1]]
    prop_name = ''
    for p_i in fact.prop_comb:
        prop_name += args.prop_names[p_i] + '_'
    prop_name = prop_name[:-1]

    pred = fact.preds[pred_i]
    pred_func_name = pred.name

    pred_name = obj_A + "_" + prop_name + "_" + pred_func_name + "_" + obj_B

    dtypes = [DataType(dt) for dt in [obj_A, obj_B]]
    pred = InvPredicate(pred_name, 2, dtypes, config.func_pred_name,
                        grounded_prop=prop_name,
                        grounded_objs=[obj_A, obj_B],
                        pred_func=pred_func_name)
    return pred


def behavior_action_as_head_atom(args, behavior):
    action_predicate = generate_action_predicate(args, behavior)
    head_atom = Atom(action_predicate, [Var("agent")])
    return head_atom


def behavior_predicate_as_func_atom(args, behavior):
    # behavior['grounded_objs'] determines terms in the clause
    func_atoms = []
    for fact in behavior.fact:
        terms = extract_fact_terms(args, fact)
        for pred_i in range(len(fact.preds)):
            func_pred = generate_func_predicate(args, fact, pred_i)
            func_atoms.append(Atom(func_pred, terms))
    return func_atoms


def behavior_existence_as_env_atoms(args, behavior):
    mask = behavior.fact[0].mask
    exist_atoms = []
    for o_i, exist_obj in enumerate(mask):

        if not exist_obj:
            existence = "not_exist"
            obj_name = args.obj_info[o_i][0]
        else:
            existence = "exist"
            obj_name = args.obj_info[o_i][0]
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
        clause = behavior2clause(args, behavior)
        clauses.append(clause)
        # clause_weights.append()
    # print(f'======= Clauses from Behaviors {len(clauses)} ======')
    # for c in clauses:
    #     print(c)
    return clauses
