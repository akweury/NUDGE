# Created by shaji at 01/01/2024
import json
import shutil
import os
import random
import torch
from torch import nn
from tqdm import tqdm
from ocatari.core import OCAtari
from gzip import GzipFile
from pathlib import Path
from functools import partial
from ale_env import ALEModern, ALEClassic

from src import config
from src.agents.smp_agent import SymbolicMicroProgramPlayer
from src.agents.random_agent import RandomPlayer
from src.environments.getout.getout.getout.getout import Getout
from src.environments.getout.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from src.utils_game import render_getout, render_assault, render_loot, render_ecoinrun, render_atari

from src.agents.utils_loot import extract_neural_state_loot, simplify_action_loot
from src.agents.utils_getout import extract_logic_state_getout


class RolloutBuffer:
    def __init__(self, filename):
        self.filename = config.path_output / 'bs_data' / filename
        self.actions = []
        self.logic_states = []
        self.neural_states = []
        self.action_probs = []
        self.logprobs = []
        self.rewards = []
        self.ungrounded_rewards = []
        self.terminated = []
        self.predictions = []
        self.reason_source = []

    def clear(self):
        del self.actions[:]
        del self.logic_states[:]
        del self.neural_states[:]
        del self.action_probs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.ungrounded_rewards[:]
        del self.terminated[:]
        del self.predictions[:]
        del self.reason_source[:]

    def load_buffer(self, args):
        with open(config.path_bs_data / args.filename, 'r') as f:
            state_info = json.load(f)
        print(f"buffer file: {args.filename}")
        self.actions = torch.tensor(state_info['actions']).to(args.device)
        self.logic_states = torch.tensor(state_info['logic_states']).to(args.device)
        self.neural_states = torch.tensor(state_info['neural_states']).to(args.device)
        self.action_probs = torch.tensor(state_info['action_probs']).to(args.device)
        self.logprobs = torch.tensor(state_info['logprobs']).to(args.device)
        self.rewards = torch.tensor(state_info['reward']).to(args.device)
        self.terminated = torch.tensor(state_info['terminated']).to(args.device)
        self.predictions = torch.tensor(state_info['predictions']).to(args.device)

        if 'ungrounded_rewards' in list(state_info.keys()):
            self.ungrounded_rewards = state_info['ungrounded_rewards']
        if "reason_source" not in list(state_info.keys()):
            self.reason_source = ["neural"] * len(self.actions)
        else:
            self.reason_source = state_info['reason_source']

    def save_data(self):
        data = {'actions': self.actions,
                'logic_states': self.logic_states,
                'neural_states': self.neural_states,
                'action_probs': self.action_probs,
                'logprobs': self.logprobs,
                'reward': self.rewards,
                'ungrounded_rewards': self.ungrounded_rewards,
                'terminated': self.terminated,
                'predictions': self.predictions,
                "reason_source": self.reason_source
                }

        with open(self.filename, 'w') as f:
            json.dump(data, f)
        print(f'data saved in file {self.filename}')


def load_buffer(args):
    buffer = RolloutBuffer(args.filename)
    buffer.load_buffer(args)
    return buffer


def _load_checkpoint(fpath, device="cpu"):
    fpath = Path(fpath)
    with fpath.open("rb") as file:
        with GzipFile(fileobj=file) as inflated:
            return torch.load(inflated, map_location=device)


class AtariNet(nn.Module):
    """ Estimator used by DQN-style algorithms for ATARI games.
        Works with DQN, M-DQN and C51.
    """

    def __init__(self, action_no, distributional=False):
        super().__init__()

        self.action_no = out_size = action_no
        self.distributional = distributional

        # configure the support if distributional
        if distributional:
            support = torch.linspace(-10, 10, 51)
            self.__support = nn.Parameter(support, requires_grad=False)
            out_size = action_no * len(self.__support)

        # get the feature extractor and fully connected layers
        self.__features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.__head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Linear(512, out_size),
        )

    def forward(self, x):
        assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div(255)

        x = self.__features(x)
        qs = self.__head(x.view(x.size(0), -1))

        if self.distributional:
            logits = qs.view(qs.shape[0], self.action_no, len(self.__support))
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.__support.expand_as(qs_probs)).sum(2)
        return qs


def _epsilon_greedy(obs, model, eps=0.001):
    if torch.rand((1,)).item() < eps:
        return torch.randint(model.action_no, (1,)).item(), None
    q_val, argmax_a = model(obs).max(1)
    return argmax_a.item(), q_val

def create_agent(args):
    #### create agent
    if args.agent == "smp":
        agent = SymbolicMicroProgramPlayer(args)
    elif args.agent == 'random':
        agent = RandomPlayer(args)
    elif args.agent == 'human':
        agent = 'human'
    elif args.agent == 'ppo':
        # game/seed/model
        args.model_path = config.path_model / 'atari' / 'model_50000000.gz'
        ckpt = _load_checkpoint(args.model_path)
        game = 'Assault'

        # set env
        ALE = ALEModern if "_modern/" in str(args.model_path) else ALEClassic
        env = ALE(
            game,
            torch.randint(100_000, (1,)).item(),
            sdl=True,
            device="cpu",
            clip_rewards_val=False,
            record_dir=None,
        )

        # init model
        model = AtariNet(env.action_space.n, distributional="C51_" in str(args.model_path))

        model.load_state_dict(ckpt["estimator_state"])
        # configure policy
        policy = partial(_epsilon_greedy, model=model, eps=0.001)

        agent = policy
    else:
        raise ValueError

    return agent


def create_getout_instance(args, seed=None):
    if args.env == 'getoutplus':
        enemies = True
    else:
        enemies = False
    # level_generator = DummyGenerator()
    getout = Getout()
    level_generator = ParameterizedLevelGenerator(enemies=enemies)
    level_generator.generate(getout, seed=seed)
    getout.render()

    return getout


def render_game(agent, args):
    if args.m == 'getout':
        render_getout(agent, args)
    elif args.m == 'Assault':
        render_assault(agent, args)
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
        return
    elif args.teacher_agent == "random":
        args.model_file = "random"
        max_states = 10000
        # play games using the random agent
        seed = random.seed() if args.seed is None else int(args.seed)
        args.filename = args.m + '_' + args.teacher_agent + '_episode_' + str(args.episode) + '.json'

        buffer = RolloutBuffer(args.filename)

        if os.path.exists(buffer.filename):
            return
        # collect data
        step = 0
        collected_states = 0
        if args.m == 'getout':
            coin_jump = create_getout_instance(args)

            # frame rate limiting
            for i in tqdm(range(max_states)):
                step += 1
                # predict actions
                if not coin_jump.level.terminated:
                    # random actions
                    action, explaining = agent.reasoning_act(coin_jump)
                    logic_state = extract_logic_state_getout(coin_jump, args).squeeze()
                    reward = coin_jump.step(action)
                    collected_states += 1
                    buffer.logic_states.append(logic_state.detach().tolist())
                    buffer.actions.append(action - 1)
                    buffer.rewards.append(reward)
                    buffer.reason_source.append(explaining)
                # start a new game
                else:
                    coin_jump = create_getout_instance(args)
            buffer.save_data()
        return
    else:
        raise ValueError("Teacher agent not exist.")


def extract_logic_state_assault(objects, args, noise=False):
    num_of_feature = 6
    num_of_object = 8
    representation = objects.level.get_representation()
    # import ipdb; ipdb.set_trace()
    extracted_states = torch.zeros((num_of_object, num_of_feature))
    for entity in representation["entities"]:
        if entity[0].name == 'PLAYER':
            extracted_states[0][0] = 1
            extracted_states[0][-2:] = entity[1:3]
            # 27 is the width of map, this is normalization
            # extracted_states[0][-2:] /= 27
        elif entity[0].name == 'KEY':
            extracted_states[1][1] = 1
            extracted_states[1][-2:] = entity[1:3]
            # extracted_states[1][-2:] /= 27
        elif entity[0].name == 'DOOR':
            extracted_states[2][2] = 1
            extracted_states[2][-2:] = entity[1:3]
            # extracted_states[2][-2:] /= 27
        elif entity[0].name == 'GROUND_ENEMY':
            extracted_states[3][3] = 1
            extracted_states[3][-2:] = entity[1:3]
            # extracted_states[3][-2:] /= 27
        elif entity[0].name == 'GROUND_ENEMY2':
            extracted_states[4][3] = 1
            # extracted_states[3][-2:] /= 27
        elif entity[0].name == 'GROUND_ENEMY3':
            extracted_states[5][3] = 1
            extracted_states[5][-2:] = entity[1:3]
        elif entity[0].name == 'BUZZSAW1':
            extracted_states[6][3] = 1
            extracted_states[6][-2:] = entity[1:3]
        elif entity[0].name == 'BUZZSAW2':
            extracted_states[7][3] = 1
            extracted_states[7][-2:] = entity[1:3]

    if sum(extracted_states[:, 1]) == 0:
        key_picked = True
    else:
        key_picked = False

    def simulate_prob(extracted_states, num_of_objs, key_picked):
        for i, obj in enumerate(extracted_states):
            obj = add_noise(obj, i, num_of_objs)
            extracted_states[i] = obj
        if key_picked:
            extracted_states[:, 1] = 0
        return extracted_states

    def add_noise(obj, index_obj, num_of_objs):
        mean = torch.tensor(0.2)
        std = torch.tensor(0.05)
        noise = torch.abs(torch.normal(mean=mean, std=std)).item()
        rand_noises = torch.randint(1, 5, (num_of_objs - 1,)).tolist()
        rand_noises = [i * noise / sum(rand_noises) for i in rand_noises]
        rand_noises.insert(index_obj, 1 - noise)

        for i, noise in enumerate(rand_noises):
            obj[i] = rand_noises[i]
        return obj

    if noise:
        extracted_states = simulate_prob(extracted_states, num_of_object, key_picked)
    states = torch.tensor(extracted_states, dtype=torch.float32, device="cpu").unsqueeze(0)
    return states


def collect_data_assault(agent, args):
    args.model_file = "neural"
    args.filename = args.m + '_' + args.teacher_agent + '_episode_' + str(args.episode) + '.json'
    max_states = 10000
    buffer = RolloutBuffer(args.filename)
    if os.path.exists(buffer.filename):
        return

    step = 0
    collected_states = 0
    env = OCAtari(args.m, mode="raw", hud=True, render_mode="rgb_array")
    observation, info = env.reset()
    for i in tqdm(range(max_states)):
        # step game
        step += 1
        neural_state = observation
        # logic_state = extract_logic_state_assault(env.objects, args)
        predictions = agent(neural_state)

        action = torch.argmax(predictions)
        action = simplify_action_loot(action)

        obs, reward, terminated, truncated, info = env.step(action)

        collected_states += 1
        # buffer.logic_states.append(logic_state.detach().tolist())
        buffer.actions.append(torch.argmax(predictions.detach()).tolist())
        buffer.action_probs.append(predictions.detach().tolist())
        buffer.neural_states.append(neural_state.tolist())
    buffer.save_data()


def collect_data_game(agent, args):
    if args.m == 'getout':
        collect_data_getout(agent, args)
    elif args.m == 'Assault':
        collect_data_assault(agent, args)
    else:
        raise ValueError
