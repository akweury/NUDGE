# Created by jing at 01.12.23

import argparse
from src.utils import make_deterministic
from src import config
from pi.utils import log_utils


def load_args():
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

    args = parser.parse_args()
    if args.device != "cpu":
        args.device = int(args.device)
    args.exp_name = args.m
    args.log_file = log_utils.create_log_file(config.path_log, args.exp_name)
    make_deterministic(args.seed)
    if args.m == "getout":
        args.state_names = config.state_name_getout
        args.action_names = config.action_name_getout
        args.prop_names = config.prop_name_getout
    elif args.m == "threefish":
        args.state_names = config.state_name_threefish
        args.action_names = config.action_name_threefish
        args.prop_names = config.prop_name_threefish
    else:
        raise ValueError

    return args
