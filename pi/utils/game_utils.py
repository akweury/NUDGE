# Created by jing at 18.01.24
import json
from gzip import GzipFile
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from ocatari.vision.utils import make_darker, mark_bb

from src import config


class RolloutBuffer:
    def __init__(self, filename):
        self.filename = config.path_output / 'bs_data' / filename
        self.win_rate = 0
        self.actions = []
        self.lost_actions = []
        self.logic_states = []
        self.lost_logic_states = []
        self.neural_states = []
        self.action_probs = []
        self.logprobs = []
        self.rewards = []
        self.lost_rewards = []
        self.ungrounded_rewards = []
        self.terminated = []
        self.predictions = []
        self.reason_source = []
        self.game_number = []

    def clear(self):
        del self.win_rate
        del self.actions[:]
        del self.lost_actions[:]
        del self.logic_states[:]
        del self.lost_logic_states[:]
        del self.neural_states[:]
        del self.action_probs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.lost_rewards[:]
        del self.ungrounded_rewards[:]
        del self.terminated[:]
        del self.predictions[:]
        del self.reason_source[:]
        del self.game_number[:]

    def load_buffer(self, args):
        with open(config.path_bs_data / args.filename, 'r') as f:
            state_info = json.load(f)
        print(f"- Loaded game buffer file: {args.filename}")

        self.actions = [torch.tensor(state_info['actions'][i]) for i in range(len(state_info['actions']))]
        self.lost_actions = [torch.tensor(state_info['lost_actions'][i]) for i in
                             range(len(state_info['lost_actions']))]
        self.logic_states = [torch.tensor(state_info['logic_states'][i]) for i in
                             range(len(state_info['logic_states']))]
        self.lost_logic_states = [torch.tensor(state_info['lost_logic_states'][i]) for i in
                                  range(len(state_info['lost_logic_states']))]
        self.rewards = [torch.tensor(state_info['reward'][i]) for i in range(len(state_info['reward']))]
        self.lost_rewards = [torch.tensor(state_info['lost_rewards'][i]) for i in range(len(state_info['lost_rewards']))]
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
        if 'win_rate' in list(state_info.keys()):
            self.win_rate = state_info['win_rate']
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
                'game_number': self.game_number,
                'win_rate': self.win_rate,
                "lost_actions": self.lost_actions,
                "lost_logic_states": self.lost_logic_states,
                "lost_rewards": self.lost_rewards,

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


def _epsilon_greedy(obs, model, eps=0.001):
    if torch.rand((1,)).item() < eps:
        return torch.randint(model.action_no, (1,)).item(), None
    q_val, argmax_a = model(obs).max(1)
    return argmax_a.item(), q_val


def print_atari_screen(num, args, obs, env):
    for obj in env.objects:
        x, y = obj.xy
        if x < 160 and y < 210:  # and obj.visible
            opos = obj.xywh
            ocol = obj.rgb
            sur_col = make_darker(ocol)
            mark_bb(obs, opos, color=sur_col)
        # mark_point(obs, *opos[:2], color=(255, 255, 0))

    plt.imshow(obs)
    plt.savefig(args.output_folder / f"{num}.png")
