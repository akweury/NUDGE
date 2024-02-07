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
from PIL import Image, ImageDraw
import numpy as np
import time

from pi.utils import game_utils
from pi.Player import SymbolicMicroProgramPlayer, PpoPlayer
from pi.utils.game_utils import RolloutBuffer, _load_checkpoint, _epsilon_greedy
from pi.utils.oc_utils import extract_logic_state_assault, extract_logic_state_asterix
from pi.utils import draw_utils
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
    elif agent_type == "ppo":
        agent = PpoPlayer(args)
    elif agent_type == 'pretrained':
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
    game_num = 300
    # play games using the random agent
    seed = random.seed() if args.seed is None else int(args.seed)
    args.filename = args.m + '_' + args.teacher_agent + '_episode_' + str(game_num) + '.json'

    buffer = RolloutBuffer(args.filename)

    if os.path.exists(buffer.filename):
        return
    # collect data
    step = 0
    win_count = 0
    win_rates = []
    if args.m == 'getout':
        game_env = create_getout_instance(args)

        # frame rate limiting
        for i in tqdm(range(game_num), desc=f"win counter: {win_count}"):
            step += 1
            logic_states = []
            actions = []
            rewards = []
            frame_counter = 0

            # play a game
            while not (game_env.level.terminated):
                # random actions
                action = agent.reasoning_act(game_env)
                logic_state = extract_logic_state_getout(game_env, args).squeeze()
                try:
                    reward = game_env.step(action)
                except KeyError:
                    game_env.level.terminated = True
                    game_env.level.lost = True
                    break
                if frame_counter == 0:
                    logic_states.append(logic_state.detach().tolist())
                    actions.append(action - 1)
                    rewards.append(reward)
                elif action - 1 != actions[-1] or frame_counter % 5 == 0:
                    logic_states.append(logic_state.detach().tolist())
                    actions.append(action - 1)
                    rewards.append(reward)

                frame_counter += 1

            # save game buffer
            if not game_env.level.lost:
                buffer.logic_states.append(logic_states)
                buffer.actions.append(actions)
                buffer.rewards.append(rewards)
                win_count += 1

            else:
                buffer.lost_logic_states.append(logic_states)
                buffer.lost_actions.append(actions)
                buffer.lost_rewards.append(rewards)

            win_rates.append(win_count / (i + 1e-20))
            # start a new game
            game_env = create_getout_instance(args)

        buffer.win_rates = win_rates
        buffer.save_data()

    return


def render_assault(agent, args):
    env = OCAtari(args.m, mode="vision", hud=True, render_mode='rgb_array')
    game_num = 300
    game_i = 0
    win_counter = 0
    win_rate = torch.zeros(2, game_num)
    win_rate[1, :] = agent.buffer_win_rates[:game_num]
    observation, info = env.reset()
    zoom_in = args.zoom_in
    viewer = game_utils.setup_image_viewer(env.game_name, zoom_in * observation.shape[0],
                                           zoom_in * observation.shape[1])

    # frame rate limiting
    fps = 30
    target_frame_duration = 1 / fps
    last_frame_time = 0

    explaining = {}
    while game_i < game_num:
        frame_i = 0
        terminated = False
        truncated = False
        decision_history = []
        current_lives = args.max_lives

        while not terminated or truncated:
            current_frame_time = time.time()
            # limit frame rate
            if last_frame_time + target_frame_duration > current_frame_time:
                sl = (last_frame_time + target_frame_duration) - current_frame_time
                time.sleep(sl)
                continue
            last_frame_time = current_frame_time  # save frame start time for next iteration

            if args.agent_type == "smp":
                action, explaining = agent.act(env.objects)
            elif args.agent_type == "ppo":
                action, _ = agent(env.dqn_obs)
            else:
                raise ValueError

            obs, reward, terminated, truncated, info = env.step(action)
            if info['lives'] < current_lives:
                reward = args.reward_lost_one_live
                current_lives = info['lives']

            # logging
            for beh_i in explaining['behavior_index']:
                print(f"f: {game_i}, rw: {reward}, act: {action}, behavior: {agent.behaviors[beh_i].clause}")

            # visualization
            if frame_i % 5 == 0:
                if args.render:
                    ImageDraw.Draw(Image.fromarray(obs)).text((40, 60), "", (120, 20, 20))
                    zoom_obs = game_utils.zoom_image(obs, zoom_in * observation.shape[0],
                                                     zoom_in * observation.shape[1])
                    viewer.show(zoom_obs[:, :, :3])

            if reward < 0:
                game_i += 1
                frame_i = 0
                win_rate[0, game_i] = win_counter / (game_i + 1e-20)

            elif reward > 0:
                win_counter += 1
                game_i += 1
                frame_i = 0
                win_rate[0, game_i] = win_counter / (game_i + 1e-20)

            ram = env._env.unwrapped.ale.getRAM()
            if explaining is not None:
                explaining["reward"].append(reward)
                decision_history.append(explaining)

            # finish one game
            if terminated or truncated:
                observation, info = env.reset()

            # the game total frame number has to greater than 2
            if len(decision_history) > 2 and reward < 0:
                lost_game_data = agent.revise_loss(decision_history)
                agent.update_lost_buffer(lost_game_data)
                def_behaviors = agent.reasoning_def_behaviors(use_ckp=False)
                agent.update_behaviors(None, def_behaviors, args)
                decision_history = []
                print("- revise loss finished.")
            frame_i += 1
            draw_utils.plot_line_chart(win_rate[:, :game_i], args.output_folder, ['smp', 'ppo'], title='win_rate',
                                       cla_leg=True)
        # modify and display render
    env.close()


def collect_data_assault(agent, args):
    sample_num = 500
    args.filename = args.m + '_' + args.teacher_agent + '_sample_num_' + str(sample_num) + '.json'
    buffer = RolloutBuffer(args.filename)
    if os.path.exists(buffer.filename):
        return

    game_i = 0
    collected_states = 0
    win_count = 0
    win_rates = []
    env = OCAtari(args.m, mode="vision", hud=True, render_mode="rgb_array")
    obs, info = env.reset()
    current_lives = args.max_lives
    terminated = False
    truncated = False
    for i in tqdm(range(sample_num), desc=f"Collect Data from game: Assault"):
        # step game
        dead = False
        scored = False
        logic_states = []
        actions = []
        rewards = []
        env_objs = []
        frame_i = 0
        # states for one live
        while not dead and not scored and not terminated and not truncated:
            action, _ = agent(env.dqn_obs)
            logic_state = extract_logic_state_assault(env.objects, args)
            obs_prev = obs
            obs, frame_reward, terminated, truncated, info = env.step(action)

            reward = args.zero_reward
            env_objs.append(env.objects)

            # give negative reward
            if frame_reward < 0:
                if logic_state[:, 3].sum() == 0:
                    print("")
                reward = args.reward_lost_one_live
                dead = True

            # scoring state
            elif frame_reward > 0:
                reward = args.reward_score_one_enemy
                if logic_state[:, 0].sum() == 0:
                    reward = args.reward_lost_one_live
                scored = True
                win_count += 1

            actions.append(action)
            logic_states.append(logic_state.tolist())
            rewards.append(reward)

            Image.fromarray(game_utils.zoom_image(obs, obs.shape[0] * 3, obs.shape[1] * 3)).save(
                args.output_folder / f'{i}_{frame_i}.png_{dead}_{scored}.png', "PNG")
            frame_i += 1

        win_rates.append(win_count / (i + 1 + 1e-20))
        if dead:
            buffer.lost_logic_states.append(logic_states)
            buffer.lost_actions.append(actions)
            buffer.lost_rewards.append(rewards)
        elif scored:
            buffer.logic_states.append(logic_states)
            buffer.actions.append(actions)
            buffer.rewards.append(rewards)

        if terminated or truncated:
            terminated = False
            truncated = False
            env.reset()

    buffer.win_rates = win_rates
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
