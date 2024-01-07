# Created by shaji at 01/01/2024
import json
import shutil
import os
import random
import torch
from tqdm import tqdm

from src import config
from src.agents.smp_agent import SymbolicMicroProgramPlayer
from src.agents.random_agent import RandomPlayer
from src.environments.getout.getout.getout.getout import Getout
from src.environments.getout.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from src.utils_game import render_getout, render_threefish, render_loot, render_ecoinrun, render_atari

from nsfr.nsfr.utils import extract_for_cgen_explaining


class RolloutBuffer:
    def __init__(self, filename, reason_source):
        self.filename = config.path_output / 'bs_data' / filename
        self.actions = []
        self.logic_states = []
        self.neural_states = []
        self.action_probs = []
        self.logprobs = []
        self.rewards = []
        self.terminated = []
        self.predictions = []
        self.reason_source = reason_source

    def clear(self):
        del self.actions[:]
        del self.logic_states[:]
        del self.neural_states[:]
        del self.action_probs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.terminated[:]
        del self.predictions[:]
        del self.reason_source[:]

    def load_buffer(self, args):
        with open(config.path_bs_data / args.filename, 'r') as f:
            state_info = json.load(f)
        self.actions = torch.tensor(state_info['actions']).to(args.device)
        self.logic_states = torch.tensor(state_info['logic_states']).to(args.device)
        self.neural_states = torch.tensor(state_info['neural_states']).to(args.device)
        self.action_probs = torch.tensor(state_info['action_probs']).to(args.device)
        self.logprobs = torch.tensor(state_info['logprobs']).to(args.device)
        self.rewards = torch.tensor(state_info['reward']).to(args.device)
        self.terminated = torch.tensor(state_info['terminated']).to(args.device)
        self.predictions = torch.tensor(state_info['predictions']).to(args.device)

        # self.reason_source = state_info['reason_source']

    def save_data(self):
        data = {'actions': self.actions,
                'logic_states': self.logic_states,
                'neural_states': self.neural_states,
                'action_probs': self.action_probs,
                'logprobs': self.logprobs,
                'reward': self.rewards,
                'terminated': self.terminated,
                'predictions': self.predictions,
                "reason_source": self.reason_source
                }

        with open(self.filename, 'w') as f:
            json.dump(data, f)
        print(f'data saved in file {self.filename}')


def load_buffer(args):
    buffer = RolloutBuffer(args.filename, args.teacher_agent)
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


def create_getout_instance(seed=None):
    seed = random.randint(0, 100000000) if seed is None else seed

    # level_generator = DummyGenerator()
    coin_jump = Getout(start_on_first_action=False)
    level_generator = ParameterizedLevelGenerator()

    level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def render_game(agent, args):
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


def collect_data_getout(agent, args):
    if args.teacher_agent == "neural":
        args.filename = args.m + ".json"
        if not os.path.exists(config.path_bs_data / args.filename):
            shutil.copyfile(config.path_saved_bs_data / args.filename, config.path_bs_data / args.filename)
        with open(config.path_bs_data / args.filename, 'a') as f:
            state_info = json.load(f)
            state_info.update({"reason_source": "neural"})

        return
    elif args.teacher_agent == "random":
        args.model_file = "random"
        max_states = 10000
        # play games using the random agent
        seed = random.seed() if args.seed is None else int(args.seed)
        args.filename = args.m + '_' + args.teacher_agent + '_' + str(max_states) + '.json'

        buffer = RolloutBuffer(args.filename, args.teacher_agent)

        if os.path.exists(buffer.filename):
            return
        # collect data
        step = 0
        collected_states = 0
        if args.m == 'getout':
            coin_jump = create_getout_instance(seed=seed)
            # frame rate limiting
            for i in tqdm(range(max_states)):
                step += 1
                # predict actions
                if not coin_jump.level.terminated:
                    # random actions
                    action, explaining = agent.act(coin_jump)
                    logic_state = extract_for_cgen_explaining(coin_jump)
                    reward = coin_jump.step(action)
                    # save state/actions
                    # if reward > 0:
                    collected_states += 1
                    buffer.logic_states.append(logic_state.detach().tolist())
                    buffer.actions.append(action - 1)
                    # buffer.action_probs.append(action_probs.tolist())
                    buffer.rewards.append(reward)
                    # print(f"- collected states: {collected_states}/{max_states}")
                # start a new game
                else:
                    coin_jump = create_getout_instance(seed=seed)
                    action = 0

            buffer.save_data()
        return
    else:
        raise ValueError("Teacher agent not exist.")


def collect_data_game(agent, args):
    if args.m == 'getout':
        collect_data_getout(agent, args)
    else:
        raise ValueError


def play_games(args, agent, collect_data=False, render=False):
    if render:
        render_game(agent, args)
    elif collect_data:
        collect_data_game(agent, args)
    else:
        raise ValueError
