# Created by shaji at 01/01/2024
import shutil
import os
import random
import torch
from torch import nn
from tqdm import tqdm
from ocatari.core import OCAtari
from functools import partial
from ale_env import ALEModern

from pi.Player import SymbolicMicroProgramPlayer
from pi.utils.game_utils import RolloutBuffer, _load_checkpoint, _epsilon_greedy, print_atari_screen
from pi.utils.oc_utils import extract_logic_state_assault, extract_logic_state_asterix

from src import config
from src.agents.random_agent import RandomPlayer
from src.environments.getout.getout.getout.getout import Getout
from src.environments.getout.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from src.utils_game import render_getout, render_loot, render_ecoinrun, render_atari
from src.agents.utils_getout import extract_logic_state_getout


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
        ckpt = _load_checkpoint(args.model_path)
        # set env
        env = ALEModern(
            args.m,
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
        args.filename = "getout.json"
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


def render_assault(agent, args):
    env = OCAtari(args.m, mode="vision", hud=True, render_mode='rgb_array')
    game_num = 0
    win_counter = 0
    observation, info = env.reset()
    for i in range(10000):
        if args.agent_type == "smp":
            action, _ = agent.act(env.objects)

        elif args.agent_type == "ppo":
            action, _ = agent(env.dqn_obs)
        else:
            raise ValueError
        obs, reward, terminated, truncated, info = env.step(action)
        ram = env._env.unwrapped.ale.getRAM()
        if i % 5 == 0:
            print_atari_screen(i, args, obs, env)
        print(f'Game {game_num} : Frame {i} : Action {action} : Reward {reward}')

        if terminated or truncated:
            game_num += 1
            if reward > 1:
                win_counter += 1
            print(f"Game {game_num} Win: {win_counter}/{game_num}")

            observation, info = env.reset()
        # modify and display render
    env.close()


def collect_data_assault(agent, args):
    args.model_file = "neural"
    args.filename = args.m + '_' + args.teacher_agent + '.json'
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
        # print(f'Game {game_num} : Frame {i} : Action {action} : Reward {reward}')

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


def collect_data_asterix(agent, args):
    args.model_file = "neural"
    args.filename = args.m + '_' + args.teacher_agent + '.json'
    max_states = 100000
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
        logic_state = extract_logic_state_asterix(env.objects, args)
        action, _ = agent(env.dqn_obs)
        action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        if reward < 0:
            print("Reward")
        # print(f'Game {game_num} : Frame {i} : Action {action} : Reward {reward}')

        if terminated:
            game_num += 1
            env.reset()

        collected_states += 1
        continue
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
    elif args.m == "Asterix":
        collect_data_asterix(agent, args)
    else:
        raise ValueError
