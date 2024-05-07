# Created by jing at 15.04.24
import numpy as np
from rtpt import RTPT
import time
import random
import torch
import os
import gym3


from src.environments.procgen.procgen import ProcgenGym3Env
from src.agents.utils_threefish import extract_neural_state_threefish, extract_logic_state_threefish

from tqdm import tqdm

from nesy_pi.aitk.utils.EnvArgs import EnvArgs
from nesy_pi.aitk.utils import args_utils, game_utils, draw_utils
from nesy_pi.aitk.utils.game_utils import RolloutBuffer
from src.environments.getout.getout.getout.getout import Getout
from src.environments.getout.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from src.agents.utils_getout import extract_logic_state_getout
from src.agents.neural_agent import ActorCritic
from src.utils_game import run, get_values


args = args_utils.get_args()
rtpt = RTPT(name_initials='JS', experiment_name=f"{args.m}_{args.start_frame}_{args.end_frame}",
            max_iterations=args.end_frame - args.start_frame)
# Start the RTPT tracking
rtpt.start()
# learn behaviors from data
# collect game buffer from neural agent
buffer_filename = args.trained_model_folder / f"nesy_pi_{args.teacher_game_nums}.json"


def _render(args, agent, env_args, video_out, agent_type):
    # render the game

    screen_text = (
        f"ep: {env_args.game_i}\n "
        f"act: {args.action_names[env_args.action - 1]} re: {env_args.reward}")
    # env_args.logic_state = agent.now_state
    video_out, _ = game_utils.plot_game_frame(agent_type, env_args, video_out, env_args.obs,
                                              screen_text)

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

def load_model(model_path, args, set_eval=True):
    with open(model_path, "rb") as f:
        model = ActorCritic(args).to(args.device)
        model.load_state_dict(state_dict=torch.load(f, map_location=args.device))
    if isinstance(model, ActorCritic):
        model = model.actor
        model = model.to(args.device)
        model.as_dict = True

    if set_eval:
        model = model.eval()

    return model
def collect_threefish_data(args, agent, buffer_filename, save_buffer):
    game_num = args.teacher_game_nums
    # play games using the random agent
    seed = random.seed()
    buffer = RolloutBuffer(buffer_filename)
    if os.path.exists(buffer.filename):
        return
    # collect data
    win_count = 0
    win_rates = []
    max_states = 10000
    save_frequence = 1
    step = 0
    collected_states = 0
    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array", rand_seed=args.seed, start_level=args.seed)
    if args.with_explain:
        env = gym3.ViewerWrapper(env, info_key="rgb")

    if agent == "human":
        ia = gym3.Interactive(env, info_key="rgb", height=768, width=768)
        all_summaries = run(args,buffer, ia, game_num)
        df_scores = get_values(all_summaries, "episode_return")
        data = {'reward': df_scores}

    else:
        reward, obs, done = env.observe()
        for i in tqdm(range(game_num)):
            # step game
            step += 1
            logic_states = []
            actions = []
            rewards = []
            done= False
            while not done:
                neural_state = extract_neural_state_threefish(obs, args)
                logic_state = extract_logic_state_threefish(obs, args)
                logic_state = logic_state.squeeze(0)
                action = agent.act(obs)
                # action = simplify_action_loot(action)
                env.act(action)
                rew, obs, done = env.observe()
                action = action.tolist()[0]
                # if action == 1:
                #     action = 0
                # elif action == 3:
                #     action = 1
                # elif action == 5:
                #     action = 2
                # elif action == 7:
                #     action = 3
                # else:
                #     raise ValueError
                logic_states.append(logic_state.detach().tolist())
                actions.append(action)
                rewards.append(rew.tolist()[0])

            if rewards[-1]>0:
                buffer.logic_states.append(logic_states)
                buffer.actions.append(actions)
                buffer.rewards.append(rewards)
                win_count += 1

            else:
                buffer.lost_logic_states.append(logic_states)
                buffer.lost_actions.append(actions)
                buffer.rewards.append(rewards)
            win_rates.append(win_count / (i + 1e-20))
        buffer.win_rates = win_rates
        buffer.save_data()

if not os.path.exists(buffer_filename):
    agent = game_utils.create_agent(args, agent_type=args.teacher_agent)
    collect_threefish_data(args, agent, buffer_filename, save_buffer=True)

game_buffer = game_utils.load_buffer(args, buffer_filename)
data_file = args.trained_model_folder / f"nesy_data.pth"
if not os.path.exists(data_file):
    states = torch.cat(game_buffer.logic_states, dim=0).squeeze()
    states[:, :, -2:] = (states[:, :, -2:]+3) / 27
    actions = torch.cat(game_buffer.actions, dim=0)

    new_states = []
    new_actions = []
    for s_i, state in enumerate(states):
        if len(state[:,2].unique())!=3:
            continue
        small_fish = state[state[:,2].argmin()].tolist()
        small_fish = [0,1,0, small_fish[-2], small_fish[-1]]
        big_fish = state[state[:,2].argmax()].tolist()
        big_fish = [0,0,1, big_fish[-2], big_fish[-1]]
        agent = state[0].tolist()
        agent= [1,0,0,agent[-2], agent[-1]]
        new_states.append(torch.tensor([agent, small_fish, big_fish]).unsqueeze(0))
        new_actions.append(actions[s_i].reshape(1))
    states = torch.cat(new_states, dim=0)
    actions = torch.cat(new_actions, dim=0)
    data = {}
    action_data = []
    for a_i in range(5):
        d = states[actions == a_i]
        random_indices = torch.randperm(d.shape[0])
        action_data.append(d[random_indices])
    min_size = min([len(data) for data in action_data])
    min_size = min_size - min_size % 100
    action_data = [d[:min_size] for d in action_data]

    for a_i in range(5):
        action_mask = actions == a_i
        pos_data = action_data[a_i]
        rest_data = torch.cat([d for d_i, d in enumerate(action_data) if d_i != a_i])
        neg_indices = torch.randperm(rest_data.shape[0])
        neg_data = rest_data[neg_indices][:len(pos_data)]

        data[a_i] = {"pos_data": pos_data, "neg_data": neg_data}
    torch.save(data, data_file)
    print(f"Saved data to {data_file}.")
