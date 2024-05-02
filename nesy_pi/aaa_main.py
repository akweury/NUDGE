# Created by shaji on 21-Mar-23
import os
import time
import datetime
import torch
from rtpt import RTPT

from nesy_pi.aitk.utils import log_utils, args_utils, file_utils
from nesy_pi import semantic as se
from src import config

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def init(args):
    log_utils.add_lines(f"- device: {args.device}", args.log_file)

    # Create RTPT object
    rtpt = RTPT(name_initials='JS', experiment_name=f"{args.m}_{args.start_frame}_{args.end_frame}",
                max_iterations=1000)
    # Start the RTPT tracking
    rtpt.start()
    torch.set_printoptions(precision=4)

    data_file = args.trained_model_folder / f"nesy_data.pth"
    data = torch.load(data_file, map_location=torch.device(args.device))

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
    learned_clauses = []
    clause_scores = []
    inv_consts = []
    pi_clauses = []
    inv_preds = []
    args, rtpt, data, NSFR = init(args)
    args.rule_obj_num = args.max_rule_obj

    lang = se.init_ilp(args, data, config.pi_type['bk'])
    # for a_i in range(len(args.action_names)):
    for a_i in range(len(args.action_names)):
        args.label = a_i
        args.label_name = args.action_names[a_i]
        action_clauses = []
        action_clause_scores = []
        log_utils.add_lines(
            f"============================= RULE OBJ NUM : {args.rule_obj_num} =======================",
            args.log_file)
        # set up the environment, load the dataset and results from perception models
        start = time.time()

        group_end = time.time()
        group_round_time.append(group_end - start)
        # ILP and PI system
        # lang = se.update_ilp(lang, args, data, config.pi_type['bk'])
        sorted_clauses_with_scores = se.run_ilp_train(args, lang)
        train_end = time.time()
        train_round_time.append(train_end - group_end)

        train_end = time.time()
        # se.ilp_eval(success, args, lang, clauses, g_data)
        eval_end = time.time()

        # log
        log_utils.add_lines(f"=============================", args.log_file)
        log_utils.add_lines(f"+ Grouping round time: {(sum(group_round_time) / 60):.2f} minute(s)", args.log_file)
        log_utils.add_lines(f"+ Training round time: {(sum(train_round_time) / 60):.2f} minute(s)", args.log_file)
        log_utils.add_lines(f"+ Evaluation round time: {((eval_end - train_end) / 60):.2f} minute(s)",
                            args.log_file)
        log_utils.add_lines(f"+ Running time: {((eval_end - exp_start) / 60):.2f} minute(s)", args.log_file)
        log_utils.add_lines(f"=============================", args.log_file)

        action_clauses += [cs[0] for cs in sorted_clauses_with_scores]
        action_clause_scores += [cs[1] for cs in sorted_clauses_with_scores]
        clause_scores.append(action_clause_scores)
        learned_clauses.append(action_clauses)
        action_invented_data = se.extract_invented_data(lang)

        inv_consts += action_invented_data['inv_consts']
        pi_clauses += action_invented_data['pi_clauses']
        inv_preds += action_invented_data['inv_preds']
    learned_data = {"clauses": learned_clauses,
                    "clause_scores": clause_scores,
                    "all_invented_preds": inv_preds,
                    "all_pi_clauses": pi_clauses,
                    "invented_preds": inv_preds,
                    "p_inv_counter": lang.invented_preds_number,
                    "invented_consts_number": lang.invented_consts_number,
                    "preds": lang.preds,
                    "inv_consts": inv_consts
                    }
    torch.save(learned_data,
               args.trained_model_folder / f"learned_clauses_rho{args.rho_num}_phi_{args.phi_num}_train_{args.train_data_size}.pt")

    return learned_clauses


if __name__ == "__main__":
    main()
