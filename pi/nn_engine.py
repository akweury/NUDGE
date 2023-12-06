# Created by jing at 04.12.23
import os
import datetime
import torch
from torch.optim import SGD, Adam, SparseAdam
from torch.utils.data import Dataset, DataLoader
from rtpt import RTPT
import wandb
import random

from src import config




class WeightDataset(Dataset):
    """
    dataset used for training position encoder
    """

    def __init__(self, dataset, device):
        self.data = dataset
        self.device = device

    def __getitem__(self, item):
        data = self.data[item]

        return data

    def __len__(self):
        return len(self.data)




def init_env(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='JS', experiment_name=args.exp, max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()
    if args.wandb:
        # start the wandb tracking
        wandb.init(project=f"{args.exp}", config={"learning_rate": args.lr, "epochs": args.epochs},
                   name=f"obj_{args.obj_num}_epochs_{args.epochs}_")

def split_dataset(data):
    train_size = int(0.8 * data.action_probs.size(0))
    test_size = data.action_probs.size(0) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(list(range(data.action_probs.size(0))), [train_size, test_size])


    return train_dataset, test_dataset




def create_weight_dataset(args, data):
    dataset = WeightDataset(data, args.device)
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size)

    return dataset_loader


def init_weight_dataset_from_game_buffer(args, buffer):
    buffer_train, buffer_test = split_dataset(buffer)
    # Load dataset as tensors
    data_train = create_weight_dataset(args, buffer_train)
    data_test = create_weight_dataset(args, buffer_test)
    return data_train, data_test