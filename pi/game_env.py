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
        self.game_number = []

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
        del self.game_number[:]

    def load_buffer(self, args):
        with open(config.path_bs_data / args.filename, 'r') as f:
            state_info = json.load(f)
        print(f"buffer file: {args.filename}")

        self.actions = torch.tensor(state_info['actions']).to(args.device)
        self.logic_states = torch.tensor(state_info['logic_states']).to(args.device).squeeze()
        self.rewards = torch.tensor(state_info['reward']).to(args.device)

        if 'neural_states' in list(state_info.keys()):
            self.neural_states = torch.tensor(state_info['neural_states']).to(args.device)
        if 'action_probs' in list(state_info.keys()):
            self.action_probs = torch.tensor(state_info['action_probs']).to(args.device)
        if 'logprobs' in list(state_info.keys()):
            self.logprobs = torch.tensor(state_info['logprobs']).to(args.device)
        if 'terminated' in list(state_info.keys()):
            self.terminated = torch.tensor(state_info['terminated']).to(args.device)
        if 'predictions' in list(state_info.keys()):
            self.predictions = torch.tensor(state_info['predictions']).to(args.device)
        if 'ungrounded_rewards' in list(state_info.keys()):
            self.ungrounded_rewards = state_info['ungrounded_rewards']
        if 'game_number' in list(state_info.keys()):
            self.game_number = state_info['game_number']

        if "reason_source" in list(state_info.keys()):
            self.reason_source = state_info['reason_source']
        else:
            self.reason_source = ["neural"] * len(self.actions)

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
                "reason_source": self.reason_source,
                'game_number': self.game_number
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


def create_agent(args, agent_type):
    #### create agent
    if agent_type == "smp":
        agent = SymbolicMicroProgramPlayer(args)
    elif agent_type == 'random':
        agent = RandomPlayer(args)
    elif agent_type == 'human':
        agent = 'human'
    elif agent_type == 'ppo':
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
    if args.m == 'getoutplus':
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
    if args.m == 'getout' or args.m == "getoutplus":
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
        args.filename ="getout.json"
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
    extracted_states = {'Player': {'name': 'Player', 'exist': False, 'x': [], 'y': []},
                        'PlayerMissileVertical': {'name': 'PlayerMissileVertical', 'exist': False, 'x': [], 'y': []},
                        'PlayerMissileHorizontal': {'name': 'PlayerMissileHorizontal', 'exist': False, 'x': [],
                                                    'y': []},
                        'EnemyMissile': {'name': 'EnemyMissile', 'exist': False, 'x': [], 'y': []},
                        'Enemy': {'name': 'Enemy', 'exist': False, 'x': [], 'y': []}
                        }
    # import ipdb; ipdb.set_trace()
    for object in objects:
        if object.category == 'Player':
            extracted_states['Player']['exist'] = True
            extracted_states['Player']['x'].append(object.x)
            extracted_states['Player']['y'].append(object.y)
            # 27 is the width of map, this is normalization
            # extracted_states[0][-2:] /= 27
        elif object.category == 'PlayerMissileVertical':
            extracted_states['PlayerMissileVertical']['exist'] = True
            extracted_states['PlayerMissileVertical']['x'].append(object.x)
            extracted_states['PlayerMissileVertical']['y'].append(object.y)
        elif object.category == 'PlayerMissileHorizontal':
            extracted_states['PlayerMissileHorizontal']['exist'] = True
            extracted_states['PlayerMissileHorizontal']['x'].append(object.x)
            extracted_states['PlayerMissileHorizontal']['y'].append(object.y)
        elif object.category == 'Enemy':
            extracted_states['Enemy']['exist'] = True
            extracted_states['Enemy']['x'].append(object.x)
            extracted_states['Enemy']['y'].append(object.y)
        elif object.category == 'EnemyMissile':
            extracted_states['EnemyMissile']['exist'] = True
            extracted_states['EnemyMissile']['x'].append(object.x)
            extracted_states['EnemyMissile']['y'].append(object.y)
        elif object.category == "MotherShip":
            pass
        elif object.category == "PlayerScore":
            pass
        elif object.category == "Health":
            pass
        elif object.category == "Lives":
            pass
        else:
            raise ValueError
    player_id = 0
    player_missile_vertical_id = 1
    player_missile_horizontal_id = 3
    enemy_id = 5
    enemy_missile_id = 10

    player_exist_id = 0
    player_missile_vertical_exist_id = 1
    player_missile_horizontal_exist_id = 2
    enemy_exist_id = 3
    enemy_missile_exist_id = 4
    x_idx = 5
    y_idx = 6

    states = torch.zeros((12, 7))
    if extracted_states['Player']['exist']:
        states[player_id, player_exist_id] = 1
        assert len(extracted_states['Player']['x']) == 1
        states[player_id, x_idx] = extracted_states['Player']['x'][0]
        states[player_id, y_idx] = extracted_states['Player']['y'][0]

    if extracted_states['PlayerMissileVertical']['exist']:
        for i in range(len(extracted_states['PlayerMissileVertical']['x'])):
            states[player_missile_vertical_id + i, player_missile_vertical_exist_id] = 1
            states[player_missile_vertical_id + i, x_idx] = extracted_states['PlayerMissileVertical']['x'][i]
            states[player_missile_vertical_id + i, y_idx] = extracted_states['PlayerMissileVertical']['y'][i]
            if i > 1:
                raise ValueError
    if extracted_states['PlayerMissileHorizontal']['exist']:
        for i in range(len(extracted_states['PlayerMissileHorizontal']['x'])):
            states[player_missile_horizontal_id+i, player_missile_horizontal_exist_id] = 1
            states[player_missile_horizontal_id+i, x_idx] = extracted_states['PlayerMissileHorizontal']['x'][i]
            states[player_missile_horizontal_id+i, y_idx] = extracted_states['PlayerMissileHorizontal']['y'][i]
            if i > 1:
                raise ValueError

    if extracted_states['Enemy']['exist']:
        for i in range(len(extracted_states['Enemy']['x'])):
            states[enemy_id + i, enemy_exist_id] = 1
            states[enemy_id + i, x_idx] = extracted_states['Enemy']['x'][i]
            states[enemy_id + i, y_idx] = extracted_states['Enemy']['y'][i]
            if i > 5:
                raise ValueError
    if extracted_states['EnemyMissile']['exist']:
        for i in range(len(extracted_states['EnemyMissile']['x'])):
            states[enemy_missile_id+i, enemy_missile_exist_id] = 1
            states[enemy_missile_id+i, x_idx] = extracted_states['EnemyMissile']['x'][i]
            states[enemy_missile_id+i, y_idx] = extracted_states['EnemyMissile']['y'][i]
            if i > 1:
                raise ValueError

    return states


def collect_data_assault(agent, args):
    args.model_file = "neural"
    args.filename = args.m + '_' + args.teacher_agent  + '.json'
    max_states = 10000
    buffer = RolloutBuffer(args.filename)
    if os.path.exists(buffer.filename):
        return

    step = 0
    collected_states = 0
    game_num = 0
    env = OCAtari(args.m, mode="vision", hud=True, render_mode="rgb_array")
    observation, info = env.reset()

    for i in tqdm(range(max_states)):
        # step game
        step += 1
        logic_state = extract_logic_state_assault(env.objects, args)
        action, _ = agent(env.dqn_obs)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            game_num += 1
            env.reset()

        collected_states += 1
        # buffer.logic_states.append(logic_state.detach().tolist())
        buffer.actions.append(action)
        buffer.logic_states.append(logic_state.tolist())
        buffer.rewards.append(reward)
        buffer.reason_source.append('neural')
        buffer.game_number.append(game_num)

    buffer.save_data()


def collect_data_game(agent, args):
    if args.m == 'getout':
        collect_data_getout(agent, args)
    elif args.m == 'getoutplus':
        collect_data_getout(agent, args)
    elif args.m == 'Assault':
        collect_data_assault(agent, args)
    else:
        raise ValueError
