# Created by jing at 30.05.23
from nesy_pi.aitk import valuation, facts_converter, nsfr
from nesy_pi.aitk.utils.fol.language import Language


def get_vm(args, lang):
    vm = valuation.get_valuation_module(args, lang)
    return vm


def get_fc(args, lang, vm, e):
    fc = facts_converter.FactsConverter(args, lang, vm, e)
    return fc


def get_nsfr(args, lang, FC, clauses, train=False):
    NSFR = nsfr.get_nsfr_model(args, lang, FC, clauses, train)
    return NSFR


def get_nsfr_model(args, lang):
    clauses = lang.all_clauses
    VM = get_vm(args, lang)
    FC = get_fc(args, lang, VM, args.rule_obj_num)
    NSFR = get_nsfr(args, lang, FC, clauses, train=True)
    return NSFR


def get_pretrained_lang(args, inv_consts, pi_clauses, inv_preds):
    lang = Language(args, [], 'bk_pred', inv_consts=inv_consts)
    # update language
    lang.all_clauses = args.clauses
    lang.invented_preds_with_scores = []
    lang.all_pi_clauses = pi_clauses
    lang.all_invented_preds = inv_preds
    # update predicates
    # lang.update_bk(args.neural_preds, full_bk=True)
    lang.load_minimum(args.neural_preds)
    return lang
