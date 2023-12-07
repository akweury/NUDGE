# Created by shaji at 07/12/2023

import random
import torch
from torch.utils.data import Dataset, DataLoader


class WeightDataset(Dataset):
    """
    dataset used for training position encoder
    """

    def __init__(self, action_probs, actions, logic_states, neural_states, device):
        self.action_probs = action_probs
        self.actions = actions
        self.logic_states = logic_states
        self.neural_states = neural_states
        self.device = device

    def __getitem__(self, item):


        action_prob = torch.zeros(self.action_probs[item].size(1)).to(self.device)
        action_prob[self.actions[item]] = 1
        logic_state = self.logic_states[item]
        neural_state = self.neural_states[item]

        return action_prob, logic_state, neural_state

    def __len__(self):
        return len(self.action_probs)


def split_dataset(data):
    data_size = data.action_probs.size(0)
    indices = list(range(data_size))
    random.shuffle(indices)
    train_indices = indices[:int(data_size * 0.8)]
    test_indices = indices[int(data_size * 0.8):]

    train_data = {
        "action_probs": data.action_probs[train_indices],
        "actions": data.actions[train_indices],
        "logic_states": data.logic_states[train_indices],
        "neural_states": data.neural_states[train_indices]
    }

    test_data = {
        "action_probs": data.action_probs[test_indices],
        "actions": data.actions[test_indices],
        "logic_states": data.logic_states[test_indices],
        "neural_states": data.neural_states[test_indices]
    }

    return train_data, test_data


def create_weight_dataset(args, data):
    dataset = WeightDataset(data["action_probs"], data["actions"],
                            data["logic_states"], data["neural_states"], args.device)

    dataset_loader = DataLoader(dataset, batch_size=args.batch_size)

    return dataset_loader
