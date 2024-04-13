# Created by jing at 30.05.23
from nesy_pi.aitk import valuation, facts_converter, nsfr


def get_vm(args, lang):
    vm = valuation.get_valuation_module(args, lang)
    return vm


def get_fc(args, lang,vm, e):
    fc = facts_converter.FactsConverter(args, lang, vm, e)
    return fc


def get_nsfr(args, lang, FC, clauses, train=False):
    NSFR = nsfr.get_nsfr_model(args, lang, FC, clauses, train)
    return NSFR
