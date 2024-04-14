# Created by shaji on 21-Mar-23
import os
import time
import datetime

import torch
from rtpt import RTPT

from nesy_pi.aitk.utils import log_utils, args_utils
from nesy_pi import semantic as se
from src import config

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def init(args):
    log_utils.add_lines(f"============================= RULE OBJ NUM : {args.rule_obj_num} =======================",
                        args.log_file)

    log_utils.add_lines(f"- device: {args.device}", args.log_file)

    # Create RTPT object
    rtpt = RTPT(name_initials='JS', experiment_name=f"{args.m}_{args.start_frame}_{args.end_frame}",
                max_iterations=1000)
    # Start the RTPT tracking
    rtpt.start()
    torch.set_printoptions(precision=4)
    torch.autograd.set_detect_anomaly(True)
    data_file = args.trained_model_folder / f"nesy_data.pth"
    data = torch.load(data_file)

    # load logical representations
    args.lark_path = str(config.root / 'src' / 'lark' / 'exp.lark')
    args.lang_base_path = config.root / 'data' / 'lang'

    args.index_pos = config.score_example_index["pos"]
    args.index_neg = config.score_example_index["neg"]
    NSFR = None
    return args, rtpt, data, NSFR


def main():
    exp_start = time.time()
    args = args_utils.get_args()
    group_round_time = []
    train_round_time = []
    log_file = log_utils.create_log_file(args.trained_model_folder, "nesy_train")
    print(f"- log_file_path:{log_file}")
    args.log_file = log_file
    for a_i in range(len(args.action_names)):
        args.label = a_i
        args.label_name = args.action_names[a_i]
        for obj_num in range(1, args.max_rule_obj):
            args.rule_obj_num = obj_num
            # set up the environment, load the dataset and results from perception models
            start = time.time()
            args, rtpt, data, NSFR = init(args)
            group_end = time.time()
            group_round_time.append(group_end - start)

            # ILP and PI system
            lang = se.init_ilp(args, data, config.pi_type['bk'])
            success, clauses = se.run_ilp_train(args, lang)
            train_end = time.time()
            train_round_time.append(train_end - group_end)

            train_end = time.time()
            # evaluation
            g_data = None
            se.ilp_eval(success, args, lang, clauses, g_data)
            eval_end = time.time()

            # log
            log_utils.add_lines(f"=============================", args.log_file)
            log_utils.add_lines(f"+ Grouping round time: {(sum(group_round_time) / 60):.2f} minute(s)", args.log_file)
            log_utils.add_lines(f"+ Training round time: {(sum(train_round_time) / 60):.2f} minute(s)", args.log_file)
            log_utils.add_lines(f"+ Evaluation round time: {((eval_end - train_end) / 60):.2f} minute(s)",
                                args.log_file)
            log_utils.add_lines(f"+ Running time: {((eval_end - exp_start) / 60):.2f} minute(s)", args.log_file)
            log_utils.add_lines(f"=============================", args.log_file)

            if success:
                break


if __name__ == "__main__":
    main()
