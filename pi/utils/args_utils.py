# Created by jing at 01.12.23

import argparse
import os
import json
import datetime

import pi.game_settings
from src.utils import make_deterministic
from src import config
from pi.utils import smp_utils

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def load_args(exp_args_path, m):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", help="Seed for pytorch + env", default=0,
                        required=False, action="store", dest="seed", type=int)
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m")
    parser.add_argument("-r", "--rules", type=str)
    parser.add_argument("-l", "--log", help="record the information of games", action="store_true")
    parser.add_argument("-rec", "--record", help="record the rendering of the game", action="store_true")
    parser.add_argument("--log_file_name", help="the name of log file", required=False, dest='logfile')
    parser.add_argument("--render", help="render the game", action="store_true", dest="render")
    parser.add_argument("--analysis_play", help="render and analysis version", action="store_true", default=False)
    parser.add_argument("--with_explain", help="explain the game", action="store_true", default=False)
    parser.add_argument("--save_frame", help="save each frame as img", action="store_true")
    parser.add_argument("--revise", help="revise the loss games", action="store_true", default=False)
    parser.add_argument("--device", help="cpu or cuda", default="cpu", type=str)
    parser.add_argument('-d', '--dataset', required=False, help='the dataset to load if scoring', dest='d')
    parser.add_argument('--wandb', action="store_false")
    parser.add_argument('--exp', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer for the training (sgd or adam)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (default 0.001)")
    parser.add_argument("--skill_len_max", type=int, default=5, help="Maximum skill length (default 5 frames)")
    parser.add_argument('--lr_scheduler', default="100,1000", type=str, help='lr schedular.')
    parser.add_argument("--net_name", type=str, help="The name of the neural network")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to infer with")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of Workers simultaneously putting data into RAM")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from previous work")
    parser.add_argument("--print_freq", type=int, default=100, help="Frequency of printing")
    parser.add_argument("--eval_loss_best", type=float, default=1e+20, help="Best up-to-date evaluation loss")
    parser.add_argument("--rectify_num", type=int, default=5, help="Repeat times of smp rectification.")
    parser.add_argument("--teacher_agent", type=str, default="pretrained", help="Type of the teacher agent.")
    parser.add_argument("--episode_num", type=int, default=5, help="Number of episodes to update the agent.")
    parser.add_argument("--dqn_a_episode_num", type=int, default=10000, help="Number of episodes to update the agent.")
    parser.add_argument("--dqn_c_episode_num", type=int, default=10000, help="Number of episodes to update the agent.")
    parser.add_argument("--dqn_t_episode_num", type=int, default=10000, help="Number of episodes to update the agent.")

    parser.add_argument("--zoom_in", type=int, default=2.5, help="Zoom in percentage of the game window.")
    parser.add_argument("--train_state_num", type=int, default=100000, help="Zoom in percentage of the game window.")
    parser.add_argument("--hardness", type=int, default=0, help="Hardness of the game.")
    parser.add_argument("--teacher_game_nums", type=int, default=100, help="Number of the teacher game.")
    parser.add_argument("--student_game_nums", type=int, default=1000, help="Number of the student game.")
    parser.add_argument("--train_epochs", type=int, default=50000, help="Epochs for training the predicate weight.")
    parser.add_argument("--fact_conf", type=float, default=0.1,
                        help="Minimum confidence required to save a fact as a behavior.")
    args = parser.parse_args()

    if m is not None:
        args.m = m
    if args.device != "cpu":
        args.device = int(args.device)
    # load args from json file
    args_file = config.path_args / f"{args.m}.json"
    load_args_from_file(str(args_file), args)

    if args.device != "cpu":
        args.device = int(args.device)
    args.exp_name = args.m
    # args.log_file = log_utils.create_log_file(config.path_log, args.exp_name)
    make_deterministic(args.seed)
    # output folder
    args.output_folder = config.path_check_point / f"{args.m}"
    args.trained_model_folder = config.path_check_point / f"{args.m}" / "trained_models"

    args.check_point_path = config.path_check_point / f"{args.m}"
    args.game_buffer_path = config.path_check_point / f"{args.m}" / "game_buffer"
    args.path_bs_data = config.path_bs_data / args.m
    args.model_path = config.path_model / args.m / 'model_50000000.gz'
    args.buffer_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.json"
    args.buffer_tensor_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.pt"
    args.o2o_data_file = args.check_point_path / "o2o" / f"pf_stats.json"
    args.o2o_behavior_file = args.check_point_path / "o2o" / f"o2o_behaviors.pkl"
    args.o2o_weight_file = args.check_point_path / "o2o" / f"predicate_weights.pkl"

    if not os.path.isdir(args.check_point_path):
        os.mkdir(str(args.check_point_path))
    if not os.path.exists(args.check_point_path / "defensive"):
        os.mkdir(str(args.check_point_path / "defensive"))
    if not os.path.exists(args.check_point_path / "attack"):
        os.mkdir(str(args.check_point_path / "attack"))
    if not os.path.exists(args.check_point_path / "skill_attack"):
        os.mkdir(str(args.check_point_path / "skill_attack"))
    if not os.path.exists(args.check_point_path / "path_finding"):
        os.mkdir(str(args.check_point_path / "path_finding"))
    if not os.path.exists(args.check_point_path / "o2o"):
        os.mkdir(str(args.check_point_path / "o2o"))
    if not os.path.exists(str(args.output_folder)):
        os.mkdir(str(args.output_folder))
    if not os.path.exists(str(args.trained_model_folder)):
        os.mkdir(str(args.trained_model_folder))
    if not os.path.exists(str(args.game_buffer_path)):
        os.mkdir(str(args.game_buffer_path))
    if not os.path.exists(str(args.path_bs_data)):
        os.mkdir(str(args.path_bs_data))
    if not os.path.exists(args.game_buffer_path / "key_frames"):
        os.mkdir(str(args.game_buffer_path / "key_frames"))
    if not os.path.exists(args.game_buffer_path / "frames"):
        os.mkdir(str(args.game_buffer_path / "frames"))
    if not os.path.exists(args.game_buffer_path / "acc_frames"):
        os.mkdir(str(args.game_buffer_path / "acc_frames"))
    if not os.path.exists(args.game_buffer_path / "lost_frames"):
        os.mkdir(str(args.game_buffer_path / "lost_frames"))

    if args.m == "getout":
        args.teacher_agent = "ppo"
        args.buffer_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.json"
        args.buffer_tensor_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.pt"
        args.zero_reward = -0.1
        args.var_th = 0.8
        args.step_dist = [0.01, -0.03]
        args.max_dist = 0.1
        args.teacher_game_nums = 300
        args.zoom_in = 1.5
        args.max_lives = 0
        args.reward_lost_one_live = -20
        args.pass_th = 0.7
        args.failed_th = 0.3
        args.att_var_th = 0.5
        args.model_path = config.path_model / args.m / 'ppo' / "ppo_.pth"
        args.action_names = config.action_name_getout
        args.prop_names = config.prop_name_getout
        args.game_info = config.game_info_getout
        args.obj_info = args.game_info["obj_info"]
        args.mile_stone_scores = [5, 10, 20, 40]

        args.obj_info = pi.game_settings.atari_obj_info(args.obj_info)
    elif args.m == "Assault" or args.m == "assault":
        args.model_path = config.path_model / args.m / 'model_50000000.gz'
        args.buffer_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.json"
        args.buffer_tensor_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.pt"
        args.train_nn_epochs = 2000
        args.zero_reward = 0.0
        args.fact_conf = 0.5
        args.action_names = config.action_name_assault
        args.prop_names = config.prop_name_assault
        args.max_lives = 4
        args.max_dist = 0.1
        args.reward_lost_one_live = -100
        args.reward_score_one_enemy = 10
        args.game_info = config.game_info_assault
        args.obj_info = args.game_info["obj_info"]
        args.obj_info = pi.game_settings.atari_obj_info(args.obj_info)
        args.var_th = 0.4
        args.reasoning_gap = 1
        args.step_dist = [0.01, -0.03]
        args.mile_stone_scores = [5, 10, 20, 40]
    elif args.m == "Pong" or args.m == "pong":
        args.model_path = config.path_model / args.m / 'model_50000000.gz'
        args.buffer_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.json"
        args.buffer_tensor_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.pt"
        args.o2o_data_file = args.check_point_path / "o2o" / f"pf_stats.json"
        args.o2o_behavior_file = args.check_point_path / "o2o" / f"o2o_behaviors.pkl"
        args.o2o_weight_file = args.check_point_path / "o2o" / f"predicate_weights.pkl"
        args.state_tensor_properties = ["dx_01", "dy_01", "va_dir", "vb_dir", "dir_ab"]
        args.prop_explain = {0: 'dx', 1: 'dy', 2: "va_dir", 3: "vb_dir", 4: 'dir_ab'}
        args.reward_gamma = 0.9
        args.reward_alignment = 0.01
        args.zero_reward = 0.0
        args.fact_conf = 0.5
        args.action_names = config.action_name_pong
        args.prop_names = config.prop_name_pong
        args.max_lives = 0
        args.max_dist = 0.1

        args.reward_lost_one_live = 0
        args.reward_score_one_enemy = 10
        args.game_info = config.game_info_pong
        args.obj_info = args.game_info["obj_info"]
        args.row_names, args.obj_data = config.get_obj_data(args.obj_info)
        args.var_th = 0.4
        args.reasoning_gap = 1
        args.step_dist = [0.01, -0.01]
        args.mile_stone_scores = [5, 10, 20, 40]
    elif args.m == "Frostbite":
        args.model_path = config.path_model / args.m / 'model_50000000.gz'
        args.buffer_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.json"
        args.buffer_tensor_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.pt"
        args.o2o_data_file = args.check_point_path / "o2o" / f"pf_stats.json"
        args.o2o_behavior_file = args.check_point_path / "o2o" / f"o2o_behaviors.pkl"
        args.o2o_weight_file = args.check_point_path / "o2o" / f"predicate_weights.pkl"
        args.state_tensor_properties = ["dx_01", "dy_01", "va_dir", "vb_dir", "dir_ab"]
        args.prop_explain = {0: 'dx', 1: 'dy', 2: "va_dir", 3: "vb_dir", 4: 'dir_ab'}
        args.train_epochs = 30
        args.jump_frames = 20
        args.reward_gamma = 0.9
        args.reward_alignment = 0.01
        args.train_nn_epochs = 2000
        args.zero_reward = 0.0
        args.fact_conf = 0.5
        args.action_names = config.action_name_frostbite
        args.prop_names = config.prop_info_frostbite
        args.max_lives = 0
        args.max_dist = 0.1
        args.reward_lost_one_live = 0
        args.reward_score_one_enemy = 10
        args.game_info = config.game_info_frostbite
        args.obj_info = args.game_info["obj_info"]
        args.obj_info = pi.game_settings.atari_obj_info(args.obj_info)
        args.var_th = 0.4
        args.reasoning_gap = 1
        args.step_dist = [0.01, -0.01]
        args.mile_stone_scores = [5, 10, 20, 40]
    elif args.m == "montezuma_revenge":
        args.o2o_data_file = args.check_point_path / "o2o" / f"pf_stats.json"
        args.o2o_behavior_file = args.check_point_path / "o2o" / f"o2o_behaviors.pkl"
        args.o2o_weight_file = args.check_point_path / "o2o" / f"predicate_weights.pkl"
        args.state_tensor_properties = ["dx_01", "dy_01", "va_dir", "vb_dir", "dir_ab"]
        args.prop_explain = {0: 'dx', 1: 'dy', 2: "va_dir", 3: "vb_dir", 4: 'dir_ab'}
        args.train_epochs = 30
        args.jump_frames = 20
        args.reward_gamma = 0.9
        args.reward_alignment = 0.01
        args.train_nn_epochs = 2000
        args.zero_reward = 0.0
        args.fact_conf = 0.5
        args.action_names = config.action_name_18

        args.max_lives = 0
        args.max_dist = 0.1
        args.reward_lost_one_live = 0
        args.reward_score_one_enemy = 10
        args.game_info = config.game_info_montezumaRevenge
        args.obj_info = args.game_info["obj_info"]
        args.obj_info = pi.game_settings.atari_obj_info(args.obj_info)
        args.var_th = 0.4
        args.reasoning_gap = 1
        args.step_dist = [0.01, -0.01]
        args.mile_stone_scores = [5, 10, 20, 40]
    elif args.m == "Asterix" or args.m == "asterix":
        args.jump_frames = 5
        args.model_path = config.path_model / args.m / 'model_50000000.gz'
        args.buffer_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.json"
        args.buffer_tensor_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.pt"
        args.train_nn_epochs = 2000
        args.zero_reward = 0.0
        args.fact_conf = 0.5
        args.action_names = config.action_name_asterix
        args.prop_names = config.prop_name_asterix
        args.max_lives = 3
        args.max_dist = 0.1
        args.reward_lost_one_live = -100
        args.reward_score_one_enemy = 10
        args.game_info = config.game_info_asterix
        args.obj_info = args.game_info["obj_info"]
        args.row_names, args.obj_data = config.get_obj_data(args.obj_info)

        args.var_th = 0.4
        args.reasoning_gap = 1
        args.step_dist = [0.01, -0.03]
        args.mile_stone_scores = [5, 10, 20, 40]
    elif args.m == "Breakout" or args.m == "breakout":
        args.model_path = config.path_model / args.m / 'model_50000000.gz'
        args.buffer_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.json"
        args.buffer_tensor_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.pt"
        args.zero_reward = 0.0
        args.fact_conf = 0.5
        args.action_names = config.action_name_breakout
        args.prop_names = config.prop_name_breakout
        args.max_lives = 3
        args.max_dist = 0.1
        args.reward_lost_one_live = -100
        args.reward_score_one_enemy = 10
        args.game_info = config.game_info_breakout
        args.obj_info = args.game_info["obj_info"]
        args.obj_info = pi.game_settings.atari_obj_info(args.obj_info)
        args.var_th = 0.4
        args.reasoning_gap = 1
        args.step_dist = [0.01, -0.03]
        args.mile_stone_scores = [5, 10, 20, 40]
    elif args.m == "Freeway" or args.m == "freeway":
        args.model_path = config.path_model / args.m / 'model_50000000.gz'
        args.buffer_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.json"
        args.buffer_tensor_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.pt"
        args.zero_reward = 0.0
        args.fact_conf = 0.5
        args.action_names = config.action_name_freeway
        args.prop_names = config.prop_name_freeway
        args.max_lives = 0
        args.max_dist = 0.1
        args.reward_lost_one_live = -100
        args.reward_score_one_enemy = 10
        args.game_info = config.game_info_freeway
        args.obj_info = args.game_info["obj_info"]
        args.obj_info = pi.game_settings.atari_obj_info(args.obj_info)
        args.var_th = 0.4
        args.reasoning_gap = 1
        args.step_dist = [0.01, -0.03]
        args.mile_stone_scores = [5, 10, 20, 40]
    elif args.m == "Kangaroo":
        args.jump_frames = 60
        args.model_path = config.path_model / args.m / 'model_50000000.gz'
        args.buffer_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.json"
        args.buffer_tensor_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.pt"
        args.train_nn_epochs = 50
        args.zero_reward = 0.0
        args.fact_conf = 0.1
        args.max_lives = 3
        args.max_dist = 0.1
        args.reward_lost_one_live = -100
        args.reward_score_one_enemy = 10
        args.var_th = 0.01
        args.skill_len_max = 8
        args.mile_stone_scores = [5, 10, 20, 40]
        args.action_names = config.action_name_18
        args.game_info = config.game_info_kangaroo
        args.obj_info = args.game_info["obj_info"]
        args.row_names, args.obj_data = config.get_obj_data(args.obj_info)
        args.same_others = config.get_same_others(args.row_names)
        args.state_tensor_properties = ["dx_01", "dy_01", "la0", "ra0", "va_dir", "vb_dir", "dir_ab"]

    elif args.m == "fishing_derby":
        args.jump_frames = 10
        args.model_path = config.path_model / args.m / 'model_50000000.gz'
        args.buffer_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.json"
        args.buffer_tensor_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.pt"
        args.train_nn_epochs = 100
        args.zero_reward = 0.0
        args.fact_conf = 0.1
        args.max_lives = 0
        args.max_dist = 0.1
        args.reward_lost_one_live = -100
        args.reward_score_one_enemy = 10
        args.var_th = 0.01
        args.step_dist = [0.01, -0.03]
        args.skill_len_max = 8

        args.mile_stone_scores = [5, 10, 20, 40]
        args.action_names = config.action_name_18
        args.prop_names = config.prop_name_kangaroo
        args.game_info = config.game_info_fishingderby
        args.obj_info = args.game_info["obj_info"]
        args.obj_info = pi.game_settings.atari_obj_info(args.obj_info)

    elif args.m == "Boxing":
        args.jump_frames = 3
        args.model_path = config.path_model / args.m / 'model_50000000.gz'
        args.buffer_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.json"
        args.buffer_tensor_filename = config.path_check_point / args.m / f"z_buffer_{str(args.teacher_agent)}_{args.teacher_game_nums}.pt"
        args.o2o_data_file = args.check_point_path / "o2o" / f"pf_stats.json"
        args.o2o_behavior_file = args.check_point_path / "o2o" / f"o2o_behaviors.pkl"
        args.o2o_weight_file = args.check_point_path / "o2o" / f"predicate_weights.pkl"
        args.train_nn_epochs = 5000

        args.reward_gamma = 0.7
        args.reward_alignment = 0.1
        args.zero_reward = 0.0
        args.fact_conf = 0.5
        args.action_names = config.action_name_boxing
        args.prop_names = config.prop_name_boxing
        args.max_lives = 0
        args.reward_lost_one_live = 0
        args.reward_score_one_enemy = 0
        args.game_info = config.game_info_boxing
        args.obj_info = args.game_info["obj_info"]
        args.obj_info = pi.game_settings.atari_obj_info(args.obj_info)
        args.var_th = 0.5
        args.skill_var_th = 0.02
        args.max_dist = 0.2
        args.skill_len_max = 8
        args.reasoning_gap = 1
        args.att_var_th = 0.1
        args.step_dist = [0.01, -0.01]
        args.mile_stone_scores = [5, 10, 20, 40]
        args.state_tensor_properties = ["dx_01", "dy_01", "la0", "ra0", "va_dir", "vb_dir", "dir_ab"]
    else:
        raise ValueError

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
