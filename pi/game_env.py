# Created by shaji at 01/01/2024
import json
import torch

from src import config
from src.agents.smp_agent import SymbolicMicroProgramPlayer
from src.agents.random_agent import RandomPlayer
from src.utils_game import render_getout, render_threefish, render_loot, render_ecoinrun, render_atari


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
        file_name = str(config.path_bs_data / args.d)
        with open(file_name, 'r') as f:
            state_info = json.load(f)

        self.actions = torch.tensor(state_info['actions']).to(args.device)
        self.logic_states = torch.tensor(state_info['logic_states']).to(args.device)
        self.neural_states = torch.tensor(state_info['neural_states']).to(args.device)
        self.action_probs = torch.tensor(state_info['action_probs']).to(args.device)
        self.logprobs = torch.tensor(state_info['logprobs']).to(args.device)
        self.rewards = torch.tensor(state_info['reward']).to(args.device)
        self.terminated = torch.tensor(state_info['terminated']).to(args.device)
        self.predictions = torch.tensor(state_info['predictions']).to(args.device)


def load_buffer(args):
    buffer = RolloutBuffer()
    buffer.load_buffer(args)
    return buffer


def create_agent(args):
    #### create agent
    if args.agent == "smp":
        agent = SymbolicMicroProgramPlayer(args)
    elif args.agent == 'random':
        agent = RandomPlayer(args)
    elif args.agent == 'human':
        agent = 'human'
    else:
        raise ValueError

    return agent


def play_games_and_render(args, agent):
    if args.m == 'getout':
        render_getout(agent, args)
    elif args.m == 'threefish':
        render_threefish(agent, args)
    elif args.m == 'loot':
        render_loot(agent, args)
    elif args.m == 'ecoinrun':
        render_ecoinrun(agent, args)
    elif args.m == 'atari':
        render_atari(agent, args)
    else:
        raise ValueError("Game not exist.")


def play_games_and_collect_data(args, agent):
    if args.teacher_agent == "neural":
        return
    elif args.teacher_agent == "random":
        # play games using the random agent
        return
    else:
        raise ValueError("Teacher agent not exist.")
