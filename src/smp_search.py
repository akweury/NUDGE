# Created by shaji at 30/11/2023

import argparse
import torch
import os
import json

from pi import micro_programs, pi_lang
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

    parser.add_argument('-m', "--model", required=True, help="the game mode for beam-search", dest='m',
                        choices=['getout', 'threefish', 'loot'])
    parser.add_argument('-d', '--dataset', required=False, help='the dataset to load if scoring', dest='d')
    parser.add_argument('--device', type=str, default="cpu")
    args = parser.parse_args()

    if args.device != "cpu":
        args.device = int(args.device)
    if args.m == "getout":
        args.state_names = config.state_name_getout
        args.action_names = config.action_name_getout
        args.prop_names = config.prop_name_getout
    elif args.m == "threefish":
        args.state_names = config.state_name_threefish
        args.action_names = config.action_name_threefish
        args.prop_names = config.prop_name_threefish
    else:
        raise ValueError

    return args


def buffer2clauses(args, buffer):
    agent_behaviors = micro_programs.buffer2behaviors(args, buffer)
    clauses = pi_lang.behaviors2clauses(args, agent_behaviors)
    return clauses


if __name__ == "__main__":
    args = get_args()

    buffer = RolloutBuffer()
    buffer.load_buffer(args)

    clauses = buffer2clauses(args, buffer)
    print("program finished!")
