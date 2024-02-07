# Created by jing at 18.01.24
import json
from gzip import GzipFile
from pathlib import Path
import torch
import cv2 as cv


from pi.utils.atari import assault_utils
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
        self.win_rates = torch.tensor(state_info['win_rates'])
        self.actions = [torch.tensor(state_info['actions'][i]) for i in range(len(state_info['actions']))]
        self.lost_actions = [torch.tensor(state_info['lost_actions'][i]) for i in
                             range(len(state_info['lost_actions']))]
        self.logic_states = [torch.tensor(state_info['logic_states'][i]) for i in
                             range(len(state_info['logic_states']))]
        self.lost_logic_states = [torch.tensor(state_info['lost_logic_states'][i]) for i in
                                  range(len(state_info['lost_logic_states']))]
        self.rewards = [torch.tensor(state_info['reward'][i]) for i in range(len(state_info['reward']))]

        self.lost_rewards = [torch.tensor(state_info['lost_rewards'][i]) for i in
                             range(len(state_info['lost_rewards']))]

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
                'game_number': self.game_number,
                'win_rates': self.win_rates,
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


def setup_image_viewer(game_name, height, width):
    viewer = assault_utils.ImageViewer(game_name, height, width, monitor_keyboard=True, )
    return viewer


def zoom_image(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
