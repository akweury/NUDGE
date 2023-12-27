# Created by jing at 01.12.23

import argparse
import os
import json
import datetime

from src.utils import make_deterministic
from src import config
from pi.utils import log_utils



date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def load_args(exp_args_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", help="Seed for pytorch + env", default=0,
                        required=False, action="store", dest="seed", type=int)
    parser.add_argument("--agent", help="agent type", required=True,
                        choices=['ppo', 'logic', 'random', 'human', "smp"])
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m",
                        choices=['getout', 'threefish', 'loot', 'ecoinrun', 'atari'])
    parser.add_argument("-env", "--environment", help="environment of game to use",
                        required=True, action="store", dest="env",
                        choices=['getout', 'getoutplus', 'getout4en',
                                 'threefish', 'threefishcolor',
                                 'loot', 'lootcolor', 'lootplus', 'loothard',
                                 'ecoinrun', 'freeway', 'kangaroo', 'asterix'])
    parser.add_argument("-r", "--rules", type=str)
    parser.add_argument("-l", "--log", help="record the information of games", action="store_true")
    parser.add_argument("-rec", "--record", help="record the rendering of the game", action="store_true")
    parser.add_argument("--log_file_name", help="the name of log file", required=False, dest='logfile')
    parser.add_argument("--render", help="render the game", action="store_true", dest="render")
    parser.add_argument("--device", help="cpu or cuda", default="cpu", type=str)
    parser.add_argument('-d', '--dataset', required=False, help='the dataset to load if scoring', dest='d')
    parser.add_argument('--wandb', action="store_false")
    parser.add_argument('--exp', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer for the training (sgd or adam)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (default 0.001)")
    parser.add_argument('--lr_scheduler', default="100,1000", type=str, help='lr schedular.')
    parser.add_argument("--net_name", type=str, help="The name of the neural network")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to infer with")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of Workers simultaneously putting data into RAM")
    parser.add_argument("--resume", type=bool, default=False, help="Resume training from previous work")
    parser.add_argument("--eval_loss_best", type=float, default=1e+20, help="Best up-to-date evaluation loss")
    parser.add_argument("--rectify_num", type=int, default=5, help="Repeat times of smp rectification.")


    args = parser.parse_args()

    # load args from json file
    args_file = exp_args_path / f"{args.exp}.json"
    load_args_from_file(str(args_file), args)

    if args.device != "cpu":
        args.device = int(args.device)
    args.exp_name = args.m
    args.log_file = log_utils.create_log_file(config.path_log, args.exp_name)
    make_deterministic(args.seed)
    if args.m == "getout":
        args.state_names = config.state_name_getout
        args.action_names = config.action_names
        args.prop_names = config.prop_name_getout
    elif args.m == "threefish":
        args.state_names = config.state_name_threefish
        args.action_names = config.action_name_threefish
        args.prop_names = config.prop_name_threefish
    else:
        raise ValueError

    # output folder
    args.output_folder = config.path_log / f"{args.exp}_{date_now}_{time_now}"
    if not os.path.exists(str(args.output_folder)):
        os.mkdir(str(args.output_folder))

    return args


def load_args_from_file(args_file_path, given_args):
    if os.path.isfile(args_file_path):
        with open(args_file_path, 'r') as fp:
            loaded_args = json.load(fp)
        # Replace given_args with the loaded default values
        for key, value in loaded_args.items():
            # if key not in ['conflict_th', 'sc_th','nc_th']:  # Do not overwrite these keys
            setattr(given_args, key, value)

        print('\n==> Args were loaded from file "{}".'.format(args_file_path))
    else:
        print('\n==> Args file "{}" was not found!'.format(args_file_path))
    return None


