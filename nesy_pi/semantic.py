# Created by jing at 30.05.23

"""
semantic implementation
"""
from nesy_pi.aitk.utils import logic_utils
from nesy_pi.aitk.utils.fol.language import Language

from nesy_pi import ilp

import nesy_pi.aitk.utils.lang_utils as lang_utils


def init_ilp(args, data, pi_type):
    logic_utils.update_args(args, data)
    lang = Language(args, [], pi_type)
    return lang


def update_ilp(lang, args, data, pi_type):
    args.invented_pred_num = lang.invented_preds_number
    args.invented_consts_number = lang.invented_consts_number
    args.p_inv_counter = lang.invented_preds_number
    lang = Language(args, [], pi_type)
    return lang


def init_eval_ilp(args, percept_dict, obj_groups, obj_avail, pi_type, level, target_clauses, inv_p_clauses):
    logic_utils.update_eval_args(args, percept_dict, obj_groups, obj_avail)
    lang = Language(args, [], pi_type, level)

    reset_lang(lang, args, level, args.neural_preds, full_bk=True)
    lang.load_invented_preds(inv_p_clauses, target_clauses)
    reset_args(args)

    lang.mode_declarations = lang_utils.get_mode_declarations(args.rule_obj_num, lang)

    return lang


# def run_ilp(args, lang, level):
#     success = ilp.ilp_main(args, lang, level)
#     return success

def clause2scores():
    pass


def scene2clauses():
    pass


# to be removed
def run_ilp_predict(args, NSFR, th, split):
    acc_val, rec_val, th_val = ilp.ilp_predict(NSFR, args, th=th, split=split)
    return acc_val, rec_val, th_val


# ---------------------------- ilp api --------------------------------------

def reset_args(args):
    ilp.reset_args(args)


def reset_lang(lang, args, level, neural_pred, full_bk):
    init_clause, e = ilp.reset_lang(lang, args, level, neural_pred, full_bk)
    return init_clause, e




def run_ilp_train(args, lang):
    ilp.ilp_train(args, lang)
    success, sorted_clauses_with_scores = ilp.ilp_test(args, lang)
    return sorted_clauses_with_scores


def ilp_robust_eval(args, lang):
    scores = ilp.ilp_robust_eval(args, lang)
    return scores


def train_nsfr(args, rtpt, lang, clauses):
    ilp.train_nsfr(args, rtpt, lang, clauses)


def extract_invented_data(lang):
    data = ilp.extract_invented_data(lang)
    return data
