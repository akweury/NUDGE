# Created by jing at 04.12.23

import os

import torch

from pi import nn_engine
from src import config


def train_clause_weights(args, buffer):
    model = nn_engine.load_model(args)
    if model is not None:
        return model

    # init tracking programs, log files
    nn_engine.init_env(args)

    # load dataset
    train_loader, test_loader = nn_engine.init_weight_dataset_from_game_buffer(args, buffer)

    # init dataset, model and its parameter settings
    model = nn_engine.init_model(args)

    for epoch in range(1, args.epochs):
        # train
        nn_engine.train_epoch(args, epoch, model, train_loader)

        # test
        nn_engine.test_epoch(args, epoch, model, test_loader)

        # save checkpoint
        nn_engine.save_checkpoint(args, epoch, model)

    return model



if __name__ == "__main__":
    train_clause_weights()


