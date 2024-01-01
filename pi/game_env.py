# Created by shaji at 01/01/2024
import json
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

    def save_data(self, args):
        dict = {'actions': self.actions, 'logic_states': self.logic_states, 'neural_states': self.neural_states,
                'action_probs': self.action_probs, 'logprobs': self.logprobs, 'reward': self.rewards,
                'terminated': self.terminated, 'predictions': self.predictions}

        dataset = args.m + '_' + args.model_file + '.json'
        path = config.path_output / 'bs_data'
        if not os.path.exists(path):
            os.mkdir(path)
        file_name = path / dataset
        with open(file_name, 'w') as f:
            json.dump(dict, f)
        print(f'data saved in file {file_name}')


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


def create_getout_instance(seed=None):
    seed = random.randint(0, 100000000) if seed is None else seed

    # level_generator = DummyGenerator()
    coin_jump = Getout(start_on_first_action=False)
    level_generator = ParameterizedLevelGenerator()

    level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def play_games_and_collect_data(args, agent):
    if args.teacher_agent == "neural":
        return
    elif args.teacher_agent == "random":
        args.model_file = "random"
        # play games using the random agent
        seed = random.seed() if args.seed is None else int(args.seed)
        buffer = RolloutBuffer()
        # collect data
        max_states = 50
        save_frequence = 5
        step = 0
        collected_states = 0
        if args.m == 'getout':
            coin_jump = create_getout_instance(seed=seed)
            # frame rate limiting
            fps = 10
            for i in tqdm(range(max_states)):
                step += 1
                # predict actions
                if not coin_jump.level.terminated:
                    action_probs = torch.rand(3)
                    action = torch.argmax(action_probs).cpu().item() + 1
                    logic_state = extract_for_cgen_explaining(coin_jump)
                    # save state/actions
                    if step % save_frequence == 0:
                        collected_states += 1
                        buffer.logic_states.append(logic_state.detach().tolist())
                        buffer.actions.append(action - 1)
                        buffer.action_probs.append(action_probs.tolist())
                # start a new game
                else:
                    coin_jump = create_getout_instance(seed=seed)
                    action = 0

                reward = coin_jump.step(action)

            buffer.save_data(args)
        return
    else:
        raise ValueError("Teacher agent not exist.")
