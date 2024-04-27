# Created by jing at 15.04.24


import time
from rtpt import RTPT
import numpy as np
from ocatari.core import OCAtari
from nesy_pi.aitk.utils import log_utils
from tqdm import tqdm
from src import config
import torch
from nesy_pi.aitk.utils import args_utils, file_utils, game_utils, draw_utils
from nesy_pi.aitk.utils.EnvArgs import EnvArgs
from nesy_pi.aitk.utils.fol.language import Language
from nesy_pi.aitk.utils.fol import bk
from nesy_pi import ilp


def load_clauses(args):
    clause_file = args.trained_model_folder / args.learned_clause_file
    data = file_utils.load_clauses(clause_file)

    args.rule_obj_num = 10
    args.p_inv_counter = data["p_inv_counter"]
    args.invented_consts_number = 10
    # load logical representations
    args.clauses = [cs for acs in data["clauses"] for cs in acs]
    scores = [clause_score.unsqueeze(0) for scores in data["clause_scores"] for clause_score in scores]
    args.clause_scores = torch.cat(scores, dim=0).to(args.device)
    args.index_pos = config.score_example_index["pos"]
    args.index_neg = config.score_example_index["neg"]
    args.lark_path = config.path_nesy / "lark" / "exp.lark"
    args.invented_pred_num = data["p_inv_counter"]
    args.invented_consts_num = data["invented_consts_number"]
    args.batch_size = 1
    args.last_refs = []
    args.found_ns = False
    bk_preds = [bk.neural_predicate_2[bk_pred_name] for bk_pred_name in args.bk_pred_names.split(",")]
    neural_preds = file_utils.load_neural_preds(bk_preds, "bk_pred")
    args.neural_preds = [neural_pred for neural_pred in neural_preds]

    lang = Language(args, [], config.pi_type['bk'], inv_consts=data["inv_consts"])
    # update language
    lang.all_clauses = args.clauses
    lang.invented_preds_with_scores = []
    lang.all_pi_clauses = data["all_pi_clauses"]
    lang.all_invented_preds = data["all_invented_preds"]
    # update predicates
    lang.update_bk(args.neural_preds, full_bk=True)

    data_file = args.trained_model_folder / f"nesy_data.pth"
    args.data = torch.load(data_file, map_location=torch.device(args.device))
    args.lang = lang


def init(args):
    rtpt = RTPT(name_initials='JS', experiment_name=f"{args.m}_{args.start_frame}_{args.end_frame}",
                max_iterations=args.end_frame - args.start_frame)
    # Start the RTPT tracking
    rtpt.start()

    agent = game_utils.create_agent(args, agent_type="clause")
    # Initialize environment
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()

    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    dqn_a_input_shape = env.observation_space.shape
    args.num_actions = env.action_space.n

    args.log_file = log_utils.create_log_file(args.trained_model_folder, "pi")
    agent.position_norm_factor = obs.shape[0]
    return env, env_args, agent, obs


def main():
    args = args_utils.get_args()
    load_clauses(args)
    # env, env_args, agent, obs = init(args)
    agent = game_utils.create_agent(args, agent_type="clause")
    data = []
    for i in range(len(args.data)):
        action_clause_ids = [c_i for c_i, c in enumerate(args.clauses) if c.head.pred.name == args.action_names[i]]

        target_preds = args.action_names[i:i + 1]
        args.batch_size = 1000
        total_size = len(args.data[i]['pos_data'][:100000])
        final_size = total_size - total_size % args.batch_size
        random_indices = torch.randperm(len(args.data[i]['pos_data']))[:final_size]
        args.test_data = args.data[i]['pos_data'][random_indices]

        img_scores, p_pos = ilp.get_clause_score(agent.NSFR, args, target_preds, 'play')
        state_action_data = img_scores[:, :, 1].permute(1, 0)
        gt_action_data = torch.zeros_like(state_action_data)
        gt_action_data[:, action_clause_ids] = state_action_data[:, action_clause_ids]
        data.append({'X':state_action_data, 'y': gt_action_data})
    torch.save(data, args.trained_model_folder / f"strategy_data.pth")

if __name__ == "__main__":
    main()
