# Created by shaji at 01/01/2024
import os
import random
from tqdm import tqdm
from ocatari.core import OCAtari
from PIL import Image

from pi.utils.EnvArgs import EnvArgs
from pi.utils import game_utils
from pi.utils.game_utils import RolloutBuffer
from pi.utils.oc_utils import extract_logic_state_assault, extract_logic_state_atari
from src.environments.getout.getout.getout.getout import Getout
from src.environments.getout.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from src.agents.utils_getout import extract_logic_state_getout


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
                args.game_buffer_path / f'{i}_{frame_i}.png_{dead}_{scored}.png', "PNG")
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
    max_games = 5
    args.filename = args.m + '_' + args.teacher_agent + str(max_games) + '.json'
    buffer = RolloutBuffer(args.filename)
    if os.path.exists(buffer.filename):
        return
    win_rates = []
    win_count = 0
    env = OCAtari(args.m, mode="revised", hud=True, render_mode="rgb_array")
    obs, info = env.reset()
    env_args = EnvArgs(args=args, game_num=300, window_size=obs.shape[:2], fps=60)
    for i in tqdm(range(max_games)):
        observation, info = env.reset()
        env_args.reset_args()
        while env_args.current_lives > 0:
            env_args.action, _ = agent(env.dqn_obs)
            env_args.obs, env_args.reward, terminated, truncated, info = env.step(env_args.action)

            env_args.logic_state, env_args.state_score = extract_logic_state_atari(env.objects, args.game_info)
            if env_args.state_score > env_args.best_score:
                env_args.best_score = env_args.state_score
            # assign reward for lost one live
            if info["lives"] < env_args.current_lives:
                env_args.current_lives = info["lives"]
                env_args.score_update = True
                env_args.win_rate[0, env_args.game_i] = env_args.best_score
                env_args.rewards[-1] += env_args.reward_lost_one_live
                env_args.dead_counter += 1
                env_args.logic_states, env_args.actions, env_args.rewards = game_utils.asterix_patches(
                    env_args.logic_states, env_args.actions, env_args.rewards)
                env_args.win_rate[0, env_args.game_i] = env_args.best_score

                buffer.logic_states.append(env_args.logic_states)
                buffer.actions.append(env_args.actions)
                buffer.rewards.append(env_args.rewards)

                env_args.logic_states = []
                env_args.actions = []
                env_args.rewards = []

                env_args.game_i += 1
            else:
                buffer.logic_states.append(env_args.logic_states)
                buffer.actions.append(env_args.actions)
                buffer.rewards.append(env_args.rewards)
            env_args.update_args(env_args)
    buffer.win_rates = win_rates
    buffer.check_validation_asterix()
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
