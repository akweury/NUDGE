# Created by jing at 04.12.23

import os
import datetime
import torch
from torch.optim import SGD, Adam, SparseAdam
from rtpt import RTPT
import wandb

from pi.neural import nn_model
from pi.utils import log_utils, loss_utils
from pi.utils.dataset import split_dataset, create_weight_dataset
from src import config

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def init_env(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='JS', experiment_name=args.exp, max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()
    if args.wandb:
        # start the wandb tracking
        wandb.init(project=f"{args.exp}", config={"learning_rate": args.lr, "epochs": args.epochs},
                   name=f"epochs_{args.epochs}_")


def init_weight_dataset_from_game_buffer(args, buffer):
    buffer_train, buffer_test = split_dataset(buffer)
    # Load dataset as tensors
    data_train = create_weight_dataset(args, buffer_train)
    data_test = create_weight_dataset(args, buffer_test)
    return data_train, data_test


def init_model(args):
    # Load network
    network = net.choose_net(args.net_name, args)
    # Load model
    if args.resume:
        log_utils.add_lines(f" ------------------ Resume a training work ------------------- ", args.log_file)
        raise NotImplementedError
    else:
        log_utils.add_lines(f" ------------------ Start a new training work ------------------- ", args.log_file)
        # init model
        model = network.to(args.device)
        args.p_bound = filter(lambda p: p.requires_grad, model.p_bound())

        # init optimizer
        if args.optimizer == 'sgd':
            args.optimizer = SGD(args.p_bound, lr=args.lr, momentum=args.momentum, weight_decay=0)
        elif args.optimizer == 'adam':
            args.optimizer = Adam(args.p_bound, lr=args.lr, weight_decay=0, amsgrad=True)
        elif args.optimizer == 'sparse_adam':
            args.optimizer = SparseAdam(args.p_bound, lr=args.lr)
        else:
            raise ValueError

    return model


def train_epoch(args, epoch, model, train_loader):
    # log
    log_utils.add_lines(f"- (Train) {datetime.datetime.now().strftime('%H:%M:%S')} "
                        f"Epoch [{epoch}] "
                        f"lr={args.optimizer.param_groups[0]['lr']:.1e} "
                        f"Start at {date_now}-{time_now} "
                        f"Train loss: {float(args.eval_loss_best):.1e}", args.log_file)
    running_loss = 0

    model.train()
    for i, input_data in enumerate(train_loader):

        if args.device != "cpu":
            # Wait for all kernels to finish
            torch.cuda.synchronize()

        # Clear the gradients
        args.optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)

        if "weight" in args.exp:
            action_prob, logic_state, neural_state = input_data

            # Forward pass. Squeeze dim 0, let number of object be the batch number, the batch size is always set to 1
            pred_action_prob = model(logic_state.unsqueeze(1))
            # Compute the loss
            loss = loss_utils.action_loss(action_prob.squeeze(), pred_action_prob)

            # Backward pass
            loss.backward()
            # Update the parameters
            args.optimizer.step()
            running_loss += loss.item()
        else:
            raise ValueError
    log_utils.add_lines(f"\tTraining Loss: {running_loss}", args.log_file)

    # log
    if args.wandb:
        wandb.log({"Train loss": running_loss})


def test_epoch(args, epoch, model, test_loader):
    # log
    log_utils.add_lines(f"- (Eval) {datetime.datetime.now().strftime('%H:%M:%S')} "
                        f"Epoch [{epoch}] "
                        f"lr={args.optimizer.param_groups[0]['lr']:.1e} "
                        f"Start from {date_now} - {time_now} "
                        f"Eval loss: {float(args.eval_loss_best):.1e}", args.log_file)
    running_vloss = 0.0
    total = 0
    correct = 0
    model.eval()
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, input_data in enumerate(test_loader):
            if args.device != "cpu":
                # Wait for all kernels to finish
                torch.cuda.synchronize()
            if "weight" in args.exp:
                action_prob, logic_state, neural_state = input_data
                action = action_prob.argmax(dim=1)
                # Forward pass. Squeeze dim 0, let number of object be the batch number, the batch size is always set to 1
                out = model(logic_state.unsqueeze(1))
                # Compute the loss
                vloss = loss_utils.action_loss(action_prob.squeeze(), out)
                _, predicted = torch.max(out.data, 1)
                correct += torch.sum(predicted == action).item()
                total += action.size(0)
            else:
                raise ValueError
            running_vloss += vloss.item()

    log_utils.add_lines(f"\tEvaluation loss: {running_vloss}", args.log_file)

    log_utils.add_lines(f"\tTest Accuracy: {100 * correct // total}%", args.log_file)

    if running_vloss < args.eval_loss_best:
        args.eval_loss_best = running_vloss
        log_utils.add_lines(f"\tnew best loss: {running_vloss}", args.log_file)

    if args.wandb:
        # log
        wandb.log({"Accuracy": 100 * correct // total})
        wandb.log({"Test loss": running_vloss})

    return None


def save_checkpoint(args, epoch, model):
    checkpoint_filename = str(config.path_check_point / args.exp / f'checkpoint-{str(epoch)}.pth.tar')
    state = {'epoch': epoch, 'model': model}
    torch.save(state, checkpoint_filename)

    # remove checkpoint in last epoch
    if epoch > 0:
        prev_checkpoint_filename = str(config.path_check_point / args.exp / f'checkpoint-{str(epoch - 1)}.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

    return None


def load_model(args):
    model_folder = config.path_check_point / args.exp
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    model_file = model_folder / config.model_name_weight
    if os.path.exists(model_file):
        model_dict = torch.load(model_file)
        return model_dict["model"]
    else:
        return None
