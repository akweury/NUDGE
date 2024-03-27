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

from pi.utils import game_utils, draw_utils, math_utils, reason_utils, file_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils import args_utils
from src import config
from pi.utils.atari import game_patches
from pi.utils.oc_utils import extract_logic_state_atari

BATCH_SIZE = 32
MEMORY_SIZE = 1000000
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.00025
TARGET_UPDATE_FREQ = 5
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
        self.num_actions = num_actions

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPSILON_MIN + (EPSILON - EPSILON_MIN) * \
                        np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long).to(state.device)

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


class Classifier(nn.Module):
    def __init__(self, num_actions):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_tensor.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_actions)  # Output size is 8 for 8 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_nn(num_actions, input_tensor, target_tensor, obj_type):
    # Define your neural network architecture

    # Instantiate the model
    model = Classifier(num_actions).to(input_tensor.device)

    # Define your loss function (cross-entropy for classification)
    criterion = nn.CrossEntropyLoss()

    # Define your optimizer (e.g., SGD, Adam, etc.)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Example data - you should replace this with your actual data
    # Input tensor shape: [batch_size, 16]
    # Target tensor shape: [batch_size]
    # Training loop
    num_epochs = 100000
    losses = torch.zeros(1, num_epochs)
    for epoch in tqdm(range(num_epochs), desc=f"obj type {obj_type}"):
        # Forward pass
        outputs = model(input_tensor)

        # Compute loss
        loss = criterion(outputs, target_tensor)

        # Zero gradients, backward pass, and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[0, epoch] = loss.detach()
        # print(f"loss {loss}")

    draw_utils.plot_line_chart(losses, path=args.output_folder, labels=["loss"], title=f"{obj_type}",
                               figure_size=(30, 5))
    return model


import os.path
from pi.utils.game_utils import create_agent
from pi.game_render import collect_data_dqn_a

args = args_utils.load_args(config.path_exps, None)

# learn behaviors from data
student_agent = create_agent(args, agent_type='smp')
# collect game buffer from neural agent
if not os.path.exists(args.buffer_filename):
    teacher_agent = create_agent(args, agent_type=args.teacher_agent)
    collect_data_dqn_a(teacher_agent, args, save_buffer=True)
student_agent.load_atari_buffer(args)

if args.m == "Pong":
    pos_data, actions = student_agent.pong_reasoner()
    num_obj_types = 2
if args.m == "Asterix":
    pos_data, actions = student_agent.asterix_reasoner()
    num_obj_types = 2
if args.m == "Kangaroo":
    pos_data, actions = student_agent.kangaroo_reasoner()
    num_obj_types = 8
# Initialize environment
env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
obs, info = env.reset()
num_actions = env.action_space.n
input_shape = env.observation_space.shape

obj_type_models = []
for obj_type in range(len(pos_data)):
    input_tensor = pos_data[obj_type].to(args.device)
    input_tensor = input_tensor.view(input_tensor.size(0), -1)
    target_tensor = actions.to(args.device)
    act_pred_model_file = args.output_folder / f"{args.m}_{obj_type}.pth.tar"

    if not os.path.exists(act_pred_model_file):
        action_pred_model = train_nn(num_actions, input_tensor, target_tensor, obj_type)
        state = {'model': action_pred_model}
        torch.save(state, act_pred_model_file)
    else:
        action_pred_model = torch.load(act_pred_model_file)["model"]
    obj_type_models.append(action_pred_model)

# Initialize agent
agent = DQNAgent(args, input_shape, num_obj_types)
agent.agent_type = "pretrained"
env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
env_args.win_rate = torch.zeros(3000)
agent.learn_performance = []
if args.with_explain:
    video_out = game_utils.get_game_viewer(env_args)
for game_i in tqdm(range(3000), desc=f"Agent  {agent.agent_type}"):
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

        obj_id = agent.select_action(env.dqn_obs.to(env_args.device))
        logic_state, _ = extract_logic_state_atari(args, env.objects, args.game_info, obs.shape[0])
        env_args.past_states.append(logic_state)
        if args.m == "Asterix":
            action = reason_utils.pred_asterix_action(args, env_args, env_args.past_states, obj_id + 1,
                                                      obj_type_models[obj_id]).to(torch.int64).reshape(1)
        elif args.m == "Pong":
            action = reason_utils.pred_pong_action(args, env_args, env_args.past_states, obj_id + 1,
                                                   obj_type_models[obj_id]).to(torch.int64).reshape(1)
        elif args.m == "Kangaroo":
            action = reason_utils.pred_kangaroo_action(args, env_args, env_args.past_states, obj_id + 1,
                                                       obj_type_models[obj_id]).to(torch.int64).reshape(1)
        else:
            raise ValueError

        state = env.dqn_obs.to(args.device)
        env_args.obs, env_args.reward, env_args.terminated, env_args.truncated, info = env.step(action)
        game_patches.atari_frame_patches(args, env_args, info)
        if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
            game_patches.atari_patches(args, agent, env_args, info)
            env_args.frame_i = len(env_args.logic_states) - 1
            env_args.update_lost_live(args.m, info["lives"])
            if sum(env_args.rewards) > 0:
                print('')
        if args.with_explain:
            screen_text = (
                f"dqn_obj ep: {env_args.game_i}, Rec: {env_args.best_score} \n "
                f"obj: {args.row_names[obj_id + 1]}, act: {args.action_names[action]} re: {env_args.reward}")
            # Red
            env_args.obs[:10, :10] = 0
            env_args.obs[:10, :10, 0] = 255
            # Blue
            env_args.obs[:10, 10:20] = 0
            env_args.obs[:10, 10:20, 2] = 255
            draw_utils.addCustomText(env_args.obs, f"dqn_obj",
                                     color=(255, 255, 255), thickness=1, font_size=0.3, pos=[1, 5])
            game_plot = draw_utils.rgb_to_bgr(env_args.obs)
            screen_plot = draw_utils.image_resize(game_plot,
                                                  int(game_plot.shape[0] * env_args.zoom_in),
                                                  int(game_plot.shape[1] * env_args.zoom_in))
            draw_utils.addText(screen_plot, screen_text,
                               color=(255, 228, 181), thickness=2, font_size=0.6, pos="upper_right")
            video_out = draw_utils.write_video_frame(video_out, screen_plot)
        # update game args
        env_args.update_args()
        env_args.rewards.append(env_args.reward)
        env_args.reward = torch.tensor(env_args.reward).reshape(1).to(args.device)
        next_state = env.dqn_obs.to(args.device) if not env_args.terminated else None
        # Store the transition in memory
        agent.memory.push(state, obj_id, next_state, env_args.reward, env_args.terminated)
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
    if game_i > args.print_freq and game_i % args.print_freq == 1:
        env_args.learn_performance.append(sum(env_args.win_rate[game_i - args.print_freq:game_i]))

        line_chart_data = torch.tensor(env_args.learn_performance)
        draw_utils.plot_line_chart(line_chart_data.unsqueeze(0), path=args.output_folder,
                                   labels=[f"total_score_every_{args.print_freq}"],
                                   title=f"{args.m}_sum_past_{args.print_freq}",
                                   figure_size=(30, 5))
        # save model
        last_epoch_save_path = args.output_folder / f'{args.m}_obj_pred_dqn_{game_i + 1 - args.print_freq}.pth'
        save_path = args.output_folder / f'{args.m}_obj_pred_dqn_{game_i + 1}.pth'
        if os.path.exists(last_epoch_save_path):
            os.remove(last_epoch_save_path)
        from pi.utils import file_utils

        file_utils.save_agent(save_path, agent)

env.close()
game_utils.finish_one_run(env_args, args, agent)
game_utils.save_game_buffer(args, env_args)
if args.with_explain:
    draw_utils.release_video(video_out)
