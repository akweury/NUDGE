# Created by shaji on 26-May-2023

import argparse
import json
import os
from src import config
from src.utils import make_deterministic


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--with_pi", action="store_true", help="Generate Clause with predicate invention.")

    parser.add_argument("--score_unique", action="store_false",
                        help="prune same score clauses.")
    parser.add_argument("--semantic_unique", action="store_false",
                        help="prune same semantic clauses.")
    parser.add_argument("--ness_maximize", action="store_true",
                        help="maximize the necessity of searched clauses.")
    parser.add_argument("--ness_equal_th", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.001, type=float,
                        help='Smooth parameter in the softor function')

    parser.add_argument("--t-beam", type=int, default=4,
                        help="Number of rule expantion of clause generation.")
    parser.add_argument("--min-beam", type=int, default=0,
                        help="The size of the minimum beam.")
    parser.add_argument("--n-beam", type=int, default=5,
                        help="The size of the beam.")
    parser.add_argument("--nc_max_step", type=int, default=3, help="The number of max steps for nc searching.")
    parser.add_argument("--max_step", type=int, default=5,
                        help="The number of max steps for clause searching.")
    parser.add_argument("--weight_tp", type=float, default=0.95,
                        help="The weight of true positive in evaluation equation.")
    parser.add_argument("--weight_length", type=float, default=0.05,
                        help="The weight of length in evaluation equation.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="The learning rate.")
    parser.add_argument("--suff_min", type=float, default=0.1,
                        help="The minimum accept threshold for sufficient clauses.")
    parser.add_argument("--sn_th", type=float, default=0.9,
                        help="The accept threshold for sufficient and necessary clauses.")
    parser.add_argument("--nc_th", type=float, default=0.9,
                        help="The accept threshold for necessary clauses.")
    parser.add_argument("--uc_th", type=float, default=0.8,
                        help="The accept threshold for unclassified clauses.")
    parser.add_argument("--sc_th", type=float, default=0.9,
                        help="The accept threshold for sufficient clauses.")
    parser.add_argument("--sn_min_th", type=float, default=0.2,
                        help="The accept sn threshold for sufficient or necessary clauses.")
    parser.add_argument("--similar_th", type=float, default=1e-3,
                        help="The minimum different requirement between any two clauses.")
    parser.add_argument("--semantic_th", type=float, default=0.75,
                        help="The minimum semantic different requirement between any two clauses.")
    parser.add_argument("--conflict_th", type=float, default=0.9,
                        help="The accept threshold for conflict clauses.")
    parser.add_argument("--length_weight", type=float, default=0.05,
                        help="The weight of clause length for clause evaluation.")
    parser.add_argument("--top_kp", type=float, default=0.95,
                        help="The accept number for clauses.")
    parser.add_argument("--uc_good_top", type=int, default=10,
                        help="The accept number for unclassified good clauses.")
    parser.add_argument("--sc_good_top", type=int, default=20,
                        help="The accept number for sufficient good clauses.")
    parser.add_argument("--sc_top", type=int, default=20,
                        help="The accept number for sufficient clauses.")
    parser.add_argument("--nc_top", type=int, default=10,
                        help="The accept number for necessary clauses.")
    parser.add_argument("--nc_good_top", type=int, default=30,
                        help="The accept number for necessary good clauses.")
    parser.add_argument("--pi_top", type=int, default=20,
                        help="The accept number for pi on each classes.")
    parser.add_argument("--max_cluster_size", type=int, default=5,
                        help="The max size of clause cluster.")
    parser.add_argument("--min_cluster_size", type=int, default=2,
                        help="The min size of clause cluster.")
    parser.add_argument("--n-data", type=float, default=200,
                        help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true",
                        help="Using pre searched clauses.")
    parser.add_argument("--top_data", type=int, default=20,
                        help="The maximum number of training data.")
    parser.add_argument("--with_bk", action="store_true",
                        help="Using background knowledge by PI.")
    parser.add_argument("--error_th", type=float, default=0.001,
                        help="The threshold for MAE of line group fitting.")
    parser.add_argument("--line_even_error", type=float, default=0.001,
                        help="The threshold for MAE of  point distribution in a line group.")
    parser.add_argument("--cir_error_th", type=float, default=0.05,
                        help="The threshold for MAE of circle group fitting.")
    parser.add_argument("--cir_even_error", type=float, default=0.001,
                        help="The threshold for MAE of point distribution in a circle group.")
    parser.add_argument("--poly_error_th", type=float, default=0.1,
                        help="The threshold for error of poly group fitting.")
    parser.add_argument("--line_group_min_sz", type=int, default=3,
                        help="The minimum objects allowed to fit a line.")
    parser.add_argument("--cir_group_min_sz", type=int, default=5,
                        help="The minimum objects allowed to fit a circle.")
    parser.add_argument("--conic_group_min_sz", type=int, default=5,
                        help="The minimum objects allowed to fit a conic section.")
    parser.add_argument("--group_conf_th", type=float, default=0.98,
                        help="The threshold of group confidence.")
    parser.add_argument("--re_eval_groups", action="store_true",
                        help="Overwrite the evaluated group detection files.")
    parser.add_argument("--maximum_obj_num", type=int, default=5,
                        help="The maximum number of objects/groups to deal with in a single image.")
    parser.add_argument("--distribute_error_th", type=float, default=0.0005,
                        help="The threshold for group points forming a shape that evenly distributed on the whole shape.")
    parser.add_argument("--show_process", action="store_false",
                        help="Print process to the logs and screen.")
    parser.add_argument("--obj_group", action="store_false",
                        help="Treat a single object as a group.")
    parser.add_argument("--line_group", action="store_false",
                        help="Treat a line of objects as a group.")
    parser.add_argument("--circle_group", action="store_false",
                        help="Treat a circle of objects as a group.")
    parser.add_argument("--conic_group", action="store_false",
                        help="Treat a conic of objects as a group.")
    parser.add_argument("--bk_pred_names", type=str,
                        help="BK predicates used in this exp.")
    parser.add_argument("--phi_num", type=int, default=20,
                        help="The number of directions for direction predicates.")
    parser.add_argument("--rho_num", type=int, default=20,
                        help="The number of distance for distance predicates.")
    parser.add_argument("--slope_num", type=int, default=10,
                        help="The number of directions for direction predicates.")
    parser.add_argument("--avg_dist_penalty", type=float, default=0.2,
                        help="The number of directions for direction predicates.")
    parser.add_argument("--cim-step", type=int, default=5,
                        help="The steps of clause infer module.")
    parser.add_argument("-s", "--seed", help="Seed for pytorch + env", default=0,
                        required=False, action="store", dest="seed", type=int)
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m")
    parser.add_argument("--learn", help="learn the invented predicates", action="store_true", default=False)
    parser.add_argument("--test", help="test the invented predicates", action="store_true", default=False)
    parser.add_argument("-r", "--rules", type=str)
    parser.add_argument( "--learned_clause_file", type=str)
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
    parser.add_argument("--player_num", type=int, default=1, help="Number of Players in the game.")
    parser.add_argument("--episode_num", type=int, default=5, help="Number of episodes to update the agent.")
    parser.add_argument("--dqn_a_episode_num", type=int, default=10000, help="Number of episodes to update the agent.")
    parser.add_argument("--dqn_c_episode_num", type=int, default=10000, help="Number of episodes to update the agent.")
    parser.add_argument("--dqn_t_episode_num", type=int, default=10000, help="Number of episodes to update the agent.")
    parser.add_argument("--stack_num", type=int, default=10, help="Zoom in percentage of the game window.")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=10000)
    parser.add_argument("--max_rule_obj", type=int, default=5)
    parser.add_argument("--ness_th", type=float, default=0.05)
    parser.add_argument("--game_obj_num", type=int)
    parser.add_argument("--zoom_in", type=int, default=2.5, help="Zoom in percentage of the game window.")
    parser.add_argument("--train_state_num", type=int, default=100000, help="Zoom in percentage of the game window.")
    parser.add_argument("--hardness", type=int, default=0, help="Hardness of the game.")
    parser.add_argument("--teacher_game_nums", type=int, default=100, help="Number of the teacher game.")
    parser.add_argument("--student_game_nums", type=int, default=10, help="Number of the student game.")
    parser.add_argument("--train_epochs", type=int, default=50000, help="Epochs for training the predicate weight.")
    parser.add_argument("--fact_conf", type=float, default=0.1,
                        help="Minimum confidence required to save a fact as a behavior.")
    args = parser.parse_args()
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
        args.model_path = config.path_model / args.m / 'Pong_ppo.cleanrl'
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
        args.row_ids, args.row_names, args.obj_data = config.get_obj_data(args.obj_info)
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
        args.row_ids, args.row_names, args.obj_data = config.get_obj_data(args.obj_info)

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
        args.model_path = config.path_model / args.m / 'Freeway_ppo.cleanrl'
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
        args.row_ids, args.row_names, args.obj_data = config.get_obj_data(args.obj_info)
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
