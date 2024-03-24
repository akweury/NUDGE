# Created by shaji at 24/03/2024

from tqdm import tqdm
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
from ocatari.core import OCAtari

from pi.utils import game_utils, draw_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils import args_utils
from src import config
from pi.utils.atari import game_patches


BATCH_SIZE = 32
MEMORY_SIZE = 1000000
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.00025
TARGET_UPDATE_FREQ = 1000
EPISODES = 1000

# Define experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define Q-network
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


# Define DQN agent
class DQNAgent:
    def __init__(self, args, input_shape, num_actions):
        self.policy_net = AtariNet(num_actions).to(args.device)
        self.target_net = AtariNet(num_actions).to(args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPSILON_MIN + (EPSILON - EPSILON_MIN) * \
                        np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long).to(state.device)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE).to(non_final_next_states.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + (next_state_values * GAMMA)

        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


args = args_utils.load_args(config.path_exps, None)

# Initialize environment
env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
obs, info = env.reset()
num_actions = env.action_space.n
input_shape = env.observation_space.shape

# Initialize agent
agent = DQNAgent(args, input_shape, num_actions)
agent.agent_type = "pretrained"
env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)

if args.with_explain:
    video_out = game_utils.get_game_viewer(env_args)
for game_i in tqdm(range(env_args.game_num), desc=f"Agent  {agent.agent_type}"):
    env_args.obs, info = env.reset()
    env_args.reset_args(game_i)
    env_args.reset_buffer_game()
    while not env_args.game_over:
        # limit frame rate
        if args.with_explain:
            current_frame_time = time.time()
            if env_args.last_frame_time + env_args.target_frame_duration > current_frame_time:
                sl = (env_args.last_frame_time + env_args.target_frame_duration) - current_frame_time
                time.sleep(sl)
                continue
            env_args.last_frame_time = current_frame_time  # save frame start time for next iteration

        action = agent.select_action(env.dqn_obs.to(env_args.device))
        state = env.dqn_obs.to(args.device)
        env_args.obs, env_args.reward, env_args.terminated, env_args.truncated, info = env.step(action)

        if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
            game_patches.atari_patches(args, agent, env_args, info)
            env_args.frame_i = len(env_args.logic_states) - 1
            env_args.update_lost_live(info["lives"])
        if args.with_explain:
            screen_text = (
                f"{agent.agent_type} ep: {env_args.game_i}, Rec: {env_args.best_score} \n "
                f"act: {args.action_names[action]} re: {env_args.reward}")
            # Red
            env_args.obs[:10, :10] = 0
            env_args.obs[:10, :10, 0] = 255
            # Blue
            env_args.obs[:10, 10:20] = 0
            env_args.obs[:10, 10:20, 2] = 255
            draw_utils.addCustomText(env_args.obs, agent.agent_type,
                                     color=(255, 255, 255), thickness=1, font_size=0.3, pos=[1, 5])
            game_plot = draw_utils.rgb_to_bgr(env_args.obs)
            screen_plot = draw_utils.image_resize(game_plot,
                                                  int(game_plot.shape[0] * env_args.zoom_in),
                                                  int(game_plot.shape[1] * env_args.zoom_in))
            draw_utils.addText(screen_plot, screen_text,
                               color=(255, 228, 181), thickness=2, font_size=0.6, pos="upper_right")

            # explain_plot_four_channel = draw_utils.three_to_four_channel(explain_plot)

            video_out = draw_utils.write_video_frame(video_out, screen_plot)
            if env_args.save_frame:
                draw_utils.save_np_as_img(screen_plot,
                                          env_args.output_folder / "frames" / f"g_{env_args.game_i}_f_{env_args.frame_i}.png")

            # game_utils.frame_log(agent, env_args)
        # update game args
        env_args.update_args()

        env_args.rewards.append(env_args.reward)
        env_args.reward = torch.tensor(env_args.reward).reshape(1).to(args.device)
        next_state = env.dqn_obs.to(args.device) if not env_args.terminated else None

        # Store the transition in memory
        agent.memory.push(state, action, next_state, env_args.reward, env_args.terminated)
        # Perform one step of optimization (on the target network)
        agent.optimize_model()
        # Update the target network, copying all weights and biases in DQN
        if game_i % TARGET_UPDATE_FREQ == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Decay epsilon
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(EPSILON_MIN, EPSILON)

    # env_args.buffer_game(args.zero_reward, args.save_frame)
    env_args.win_rate[game_i] = sum(env_args.rewards[:-1])  # update ep score
    env_args.reset_buffer_game()
    if game_i > 5:
        env_args.learn_performance[game_i] = sum(env_args.win_rate[game_i - 5:game_i])
        # print(f"win_score_sum_past_5_games: {env_args.learn_performance[game_i]}")
    if game_i % 50 == 10:
        draw_utils.plot_line_chart(env_args.learn_performance.unsqueeze(0), path=args.output_folder,
                                   labels=["sum_past_5"], title=f"{args.m}_sum_past_5_{game_i}", figure_size=(30, 5))
    # game_utils.game_over_log(args, agent, env_args)

env.close()
game_utils.finish_one_run(env_args, args, agent)
game_utils.save_game_buffer(args, env_args)
if args.with_explain:
    draw_utils.release_video(video_out)
