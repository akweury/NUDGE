import argparse
import torch
import os
import json

from nsfr.nsfr.utils_beam import get_nsfr_model
from nsfr.nsfr.logic_utils import get_lang
from nsfr.nsfr.mode_declaration import get_mode_declarations
from nsfr.nsfr.clause_generator import ClauseGenerator

from src import config


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.logic_states = []
        self.neural_states = []
        self.action_probs = []
        self.logprobs = []
        self.rewards = []
        self.terminated = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.logic_states[:]
        del self.neural_states[:]
        del self.action_probs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.terminated[:]
        del self.predictions[:]

    def load_buffer(self, args):
        current_path = os.path.dirname(__file__)
        path = os.path.join(current_path, 'bs_data', args.d)
        with open(path, 'r') as f:
            state_info = json.load(f)

        self.actions = torch.tensor(state_info['actions']).to(args.device)
        self.logic_states = torch.tensor(state_info['logic_states']).to(args.device)
        self.neural_states = torch.tensor(state_info['neural_states']).to(args.device)
        self.action_probs = torch.tensor(state_info['action_probs']).to(args.device)
        self.logprobs = torch.tensor(state_info['logprobs']).to(args.device)
        self.rewards = torch.tensor(state_info['reward']).to(args.device)
        self.terminated = torch.tensor(state_info['terminated']).to(args.device)
        self.predictions = torch.tensor(state_info['predictions']).to(args.device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int, default=1, help="Batch size in beam search")
    parser.add_argument('-r', "--rules", required=True, help="choose to root rules", dest='r',
                        choices=["getout_root", 'threefish_root', 'loot_root'])
    parser.add_argument('-m', "--model", required=True, help="the game mode for beam-search", dest='m',
                        choices=['getout', 'threefish', 'loot'])
    parser.add_argument('-t', "--t-beam", type=int, default=3, help="Number of rule expantion of clause generation.")
    parser.add_argument('-n', "--n-beam", type=int, default=8, help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50, help="The maximum number of clauses.")
    parser.add_argument("--s", type=int, default=1, help="The size of the logic program.")
    parser.add_argument('--scoring', type=bool, help='beam search rules with scored rule by trained ppo agent',
                        default=False, dest='scoring')
    parser.add_argument('-d', '--dataset', required=False, help='the dataset to load if scoring', dest='d')
    parser.add_argument('--device', type=str, default="cpu")
    args = parser.parse_args()

    if args.device != "cpu":
        args.device = int(args.device)
    if args.m == "getout":
        args.state_names = config.state_name_getout
        args.action_names = config.action_getout_dict
        args.prop_names = config.prop_name_getout
    elif args.m == "threefish":
        args.state_names = config.state_name_threefish
        args.action_names = config.action_name_threefish
        args.prop_names = config.prop_name_threefish
    else:
        raise ValueError

    return args


def run():
    args = get_args()
    # load state info for searching if scoring
    if args.scoring:
        buffer = RolloutBuffer()
        buffer.load_buffer(args)
    # writer = SummaryWriter(f"runs/{env_name}", purge_step=0)
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, '../nsfr/nsfr', 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, '../nsfr/nsfr', 'data/lang/')

    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.m, args.r)
    bk_clauses = []
    # Neuro-Symbolic Forward Reasoner for clause generation
    NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device=args.device)
    mode_declarations = get_mode_declarations(args, lang)

    print('get mode_declarations')
    if args.scoring:
        cgen = ClauseGenerator(args, NSFR_cgen, lang, atoms, mode_declarations, buffer=buffer, device=args.device)
    else:
        cgen = ClauseGenerator(args, NSFR_cgen, lang, atoms, mode_declarations, device=args.device)

    print("====== ", len(clauses), " clauses are generated!! ======")


if __name__ == "__main__":
    run()
