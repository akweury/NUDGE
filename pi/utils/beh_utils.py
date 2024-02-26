# Created by jing at 31.01.24
import torch

from pi.utils.Behavior import Behavior
from pi.utils.Fact import ProbFact, VarianceFact
from pi import predicate
from pi import pi_lang
from pi.utils import math_utils
from pi.neural import nn_model
from pi.utils import draw_utils


def create_positive_behaviors(args, pos_beh_data):
    # create positive behaviors
    path_behaviors = []

    for beh_i, beh in enumerate(pos_beh_data):
        beh_facts = []
        for fact in beh["facts"]:
            beh_facts.append(ProbFact([predicate.GT_Closest()], fact["mask"], fact["objs"], fact["props"]))
        reward = beh["expected_reward"]
        passed_state_num = beh["passed_state_num"]
        test_passed_state_num = beh["test_passed_state_num"]
        failed_state_num = beh["failed_state_num"]
        test_failed_state_num = beh["test_failed_state_num"]
        action = beh["action"]
        neg_beh = False
        behavior = Behavior("path_finding", neg_beh, beh_facts, action, reward, passed_state_num, test_passed_state_num,
                            failed_state_num, test_failed_state_num)
        behavior.clause = pi_lang.behavior2clause(args, behavior)
        path_behaviors.append(behavior)

    behavior_scores = [(beh.passed_state_num / (beh.test_passed_state_num + 1e-20)) * (
            1 - (beh.failed_state_num / (beh.test_failed_state_num + 1e-20))) for beh in path_behaviors]
    # for beh in path_behaviors:
    behavior_scores_sorted, behavior_rank = torch.tensor(behavior_scores).sort(descending=True)
    top_count = 0
    top_indices = []
    top_kp_th = 2
    for b_i, beh_score in enumerate(behavior_scores_sorted):
        top_count += beh_score
        top_indices.append(behavior_rank[b_i])
        if top_count > top_kp_th:
            break
    behaviors = [path_behaviors[i] for i in top_indices]
    for beh in behaviors:
        print(f"# path finding behavior: {beh.clause}, "
              f"++: {(beh.passed_state_num / (beh.passed_state_num + beh.failed_state_num + 1e-20)):.2f}, "
              f"+: {(beh.passed_state_num / (beh.test_passed_state_num + 1e-20)):.2f}. ")
    return path_behaviors


def create_negative_behavior(args, beh_i, beh):
    # create defense behaviors
    x_range_pos = torch.tensor(beh["x_range"], dtype=torch.float32)
    y_range_pos = torch.tensor(beh["y_range"], dtype=torch.float32)
    x_conf_pos = torch.tensor(beh["x_conf"], dtype=torch.float32)
    y_conf_pos = torch.tensor(beh["y_conf"], dtype=torch.float32)

    dir_range_pos = torch.tensor(beh["dir_range"], dtype=torch.float32)
    dir_conf_pos = torch.tensor(beh["dir_conf"], dtype=torch.float32)

    dist_pos = torch.tensor(beh["dists_pos"], dtype=torch.float32)
    dir_pos = torch.tensor(beh["dir_pos"], dtype=torch.float32)
    dist_neg = torch.tensor(beh["dists_neg"], dtype=torch.float32)
    dir_neg = torch.tensor(beh["dir_ab_neg"], dtype=torch.float32)
    pos_pos = torch.tensor(beh["position_pos"], dtype=torch.float32)
    pos_neg = torch.tensor(beh["position_neg"], dtype=torch.float32)

    expected_reward = beh["rewards"]
    obj_combs = beh["obj_combs"]
    prop_combs = beh["prop_combs"]
    action_type = beh["action_type"]
    mask = beh["masks"]

    # create predicate
    dir_names = []
    for dir in dir_range_pos:
        dir_names.append(math_utils.pol2dir_name(dir))

    pred_name = (f"dir_{dir_names}_xi{x_range_pos.min():.2f}_xa{x_range_pos.max():.2f}_"
                 f"yi_{y_range_pos.min():.2f}_ya{y_range_pos.max():.2f}")

    dist_pred = predicate.Dist_Closest(args, x_range=x_range_pos, x_conf=x_conf_pos,
                                       y_range=y_range_pos, y_conf=y_conf_pos,
                                       dir_range=dir_range_pos, dir_conf=dir_conf_pos,
                                       name=pred_name, plot_path=args.check_point_path / "defensive")
    pred = [dist_pred]

    beh_fact = VarianceFact(mask, obj_combs, prop_combs, pred)
    neg_beh = True

    behavior = Behavior("defense", neg_beh, [beh_fact], action_type, expected_reward)
    behavior.clause = pi_lang.behavior2clause(args, behavior)
    if args.with_explain:
        print(f"# defense behavior: {behavior.clause}")

    return behavior


def update_negative_behaviors(args, behaviors, def_beh_data):
    # create defense behaviors
    defense_behaviors = []
    db_plots = []
    for data_i, beh_data in enumerate(def_beh_data):
        # if behavior is exist
        behavior_exist = False
        # data_mean, data_var = beh_data['means'], beh_data['variance']
        # for beh_i, beh in enumerate(behaviors):
        #     beh_mean = behaviors[beh_i].fact[0].preds[0].mean
        #     beh_var = behaviors[beh_i].fact[0].preds[0].var
        #     if torch.abs(beh_mean - data_mean) < 1e-2 and torch.abs(beh_var - data_var) < 1e-2:
        #         print(f"- no update for behavior {behaviors[beh_i].clause}")
        #         defense_behaviors.append(behaviors[beh_i])
        #         behavior_exist = True
        #         break
        if not behavior_exist:
            behavior = create_negative_behavior(args, data_i, beh_data)
            # print(f"- new behavior {behavior.clause}")
            defense_behaviors.append(behavior)
    return defense_behaviors


def create_pf_behavior(args, beh_i, beh):
    # create attack behaviors
    x_range_pos = torch.tensor(beh["x_range"], dtype=torch.float32)
    y_range_pos = torch.tensor(beh["y_range"], dtype=torch.float32)
    x_conf_pos = torch.tensor(beh["x_conf"], dtype=torch.float32)
    y_conf_pos = torch.tensor(beh["y_conf"], dtype=torch.float32)
    dir_range_pos = torch.tensor(beh["dir_range"], dtype=torch.float32)
    dir_conf_pos = torch.tensor(beh["dir_conf"], dtype=torch.float32)

    expected_reward = beh["rewards"]
    obj_combs = beh["obj_combs"]
    prop_combs = beh["prop_combs"]
    action_type = beh["action_type"]
    mask = beh["masks"]

    # create predicate
    dir_names = []
    for dir in dir_range_pos:
        dir_names.append(math_utils.pol2dir_name(dir))
    pred_name = (f"dir_{dir_names}_xi{x_range_pos.min():.2f}_xa{x_range_pos.max():.2f}_"
                 f"yi_{y_range_pos.min():.2f}_ya{y_range_pos.max():.2f}")
    # dir_pos = torch.cat((dir_pos), dim=1)
    # dir_neg = torch.cat((dir_neg, pos_neg), dim=1)

    dist_pred = predicate.Dist_Closest(args, x_range=x_range_pos, x_conf=x_conf_pos,
                                       y_range=y_range_pos, y_conf=y_conf_pos,
                                       dir_range=dir_range_pos, dir_conf=dir_conf_pos,
                                       name=pred_name, plot_path=args.check_point_path / "path_finding")
    pred = [dist_pred]

    beh_fact = VarianceFact(mask, obj_combs, prop_combs, pred)
    neg_beh = False

    behavior = Behavior("path_finding", neg_beh, [beh_fact], action_type, expected_reward)
    behavior.clause = pi_lang.behavior2clause(args, behavior)

    print(f"# Path Finding behavior  {beh_i + 1}: {behavior.clause}")

    return behavior


def create_attack_behavior(args, beh_i, beh):
    # create attack behaviors
    x_range_pos = torch.tensor(beh["x_range"], dtype=torch.float32)
    y_range_pos = torch.tensor(beh["y_range"], dtype=torch.float32)
    x_conf_pos = torch.tensor(beh["x_conf"], dtype=torch.float32)
    y_conf_pos = torch.tensor(beh["y_conf"], dtype=torch.float32)
    dir_range_pos = torch.tensor(beh["dir_range"], dtype=torch.float32)
    dir_conf_pos = torch.tensor(beh["dir_conf"], dtype=torch.float32)

    # pos_pos = torch.tensor(beh["position_pos"], dtype=torch.float32)
    # pos_neg = torch.tensor(beh["position_neg"], dtype=torch.float32)
    expected_reward = beh["rewards"]
    obj_combs = beh["obj_combs"]
    prop_combs = beh["prop_combs"]
    action_type = beh["action_type"]
    mask = beh["masks"]

    # create predicate
    dir_names = []
    for dir in dir_range_pos:
        dir_names.append(math_utils.pol2dir_name(dir))

    pred_name = (f"dir_{dir_names}_xi{x_range_pos.min():.2f}_xa{x_range_pos.max():.2f}_"
                 f"yi_{y_range_pos.min():.2f}_ya{y_range_pos.max():.2f}")

    dist_pred = predicate.Dist_Closest(args, x_range=x_range_pos, x_conf=x_conf_pos,
                                       y_range=y_range_pos, y_conf=y_conf_pos,
                                       dir_range=dir_range_pos, dir_conf=dir_conf_pos,
                                       name=pred_name, plot_path=args.check_point_path / "attack")
    pred = [dist_pred]

    beh_fact = VarianceFact(mask, obj_combs, prop_combs, pred)
    neg_beh = False

    behavior = Behavior("attack", neg_beh, [beh_fact], action_type, expected_reward)
    behavior.clause = pi_lang.behavior2clause(args, behavior)

    print(f"# Attack behavior  {beh_i + 1}: {behavior.clause}")

    return behavior

def create_skill_attack_behavior(args, beh_i, beh):
    # create attack behaviors
    skill_len = beh["skill_len"]
    x_range_pos = [torch.tensor(beh["x_range"][i], dtype=torch.float32) for i in range(skill_len)]
    y_range_pos = [torch.tensor(beh["y_range"][i], dtype=torch.float32) for i in range(skill_len)]
    x_conf_pos = [torch.tensor(beh["x_conf"][i], dtype=torch.float32) for i in range(skill_len)]
    y_conf_pos = [torch.tensor(beh["y_conf"][i], dtype=torch.float32) for i in range(skill_len)]
    dir_range_pos = [torch.tensor(beh["dir_range"][i], dtype=torch.float32) for i in range(skill_len)]
    dir_conf_pos = [torch.tensor(beh["dir_conf"][i], dtype=torch.float32) for i in range(skill_len)]

    # pos_pos = torch.tensor(beh["position_pos"], dtype=torch.float32)
    # pos_neg = torch.tensor(beh["position_neg"], dtype=torch.float32)
    expected_reward = beh["rewards"]
    obj_combs = beh["obj_combs"]
    prop_combs = beh["prop_combs"]
    action_type = beh["action_type"]
    mask = beh["masks"]
    action_names = [args.action_names[a_i] for a_i in action_type]
    # create predicate
    dir_names = []
    for i in range(skill_len):
        dir_names.append([math_utils.pol2dir_name(dir) for dir in dir_range_pos[i]])

    pred_name = (f"skill_{action_names}")

    dist_pred = predicate.Skill_Dist_Dir(args, x_range=x_range_pos, x_conf=x_conf_pos,
                                         y_range=y_range_pos, y_conf=y_conf_pos,
                                         dir_range=dir_range_pos, dir_conf=dir_conf_pos,
                                         name=pred_name, plot_path=args.check_point_path / "skill_attack",
                                         skill_len=skill_len)
    pred = [dist_pred]

    beh_fact = VarianceFact(mask, obj_combs, prop_combs, pred)
    neg_beh = False

    behavior = Behavior("skill_attack", neg_beh, [beh_fact], action_type, expected_reward, skill_beh=True)
    behavior.clause = pi_lang.behavior2clause(args, behavior)

    print(f"# Attack behavior  {beh_i + 1}: {behavior.clause}")

    return behavior

def update_skill_attack_behaviors(args, behaviors, att_behavior_data):
    # create attack behaviors
    attack_behaviors = []
    for data_i, beh_data in enumerate(att_behavior_data):
        behavior_exist = False
        if not behavior_exist:
            behavior = create_skill_attack_behavior(args, data_i, beh_data)
            if args.with_explain:
                print(f"- new behavior {behavior.clause}")
            attack_behaviors.append(behavior)
    return attack_behaviors

def update_attack_behaviors(args, behaviors, att_behavior_data):
    # create attack behaviors
    attack_behaviors = []
    for data_i, beh_data in enumerate(att_behavior_data):
        # if behavior is exist
        behavior_exist = False
        data_mean, data_var = beh_data['means'], beh_data['variance']
        # for beh_i, beh in enumerate(behaviors):
        #     beh_mean = behaviors[beh_i].fact[0].preds[0].mean
        #     beh_var = behaviors[beh_i].fact[0].preds[0].var
        #     if torch.abs(beh_mean - data_mean) < 1e-3 and torch.abs(beh_var - data_var) < 1e-3:
        #         print(f"- no update for behavior {behaviors[beh_i].clause}")
        #         attack_behaviors.append(behaviors[beh_i])
        #         behavior_exist = True
        #         break
        if not behavior_exist:
            behavior = create_attack_behavior(args, data_i, beh_data)
            if args.with_explain:
                print(f"- new behavior {behavior.clause}")
            attack_behaviors.append(behavior)
    return attack_behaviors


def update_pf_behaviors(args, behaviors, pf_behavior_data):
    # create attack behaviors
    pf_behaviors = []
    for data_i, beh_data in enumerate(pf_behavior_data):
        # if behavior is exist
        behavior_exist = False
        # data_mean, data_var = beh_data['means'], beh_data['variance']
        # for beh_i, beh in enumerate(behaviors):
        #     beh_mean = behaviors[beh_i].fact[0].preds[0].mean
        #     beh_var = behaviors[beh_i].fact[0].preds[0].var
        #     if torch.abs(beh_mean - data_mean) < 1e-3 and torch.abs(beh_var - data_var) < 1e-3:
        #         print(f"- no update for behavior {behaviors[beh_i].clause}")
        #         pf_behaviors.append(behaviors[beh_i])
        #         behavior_exist = True
        #         break
        if not behavior_exist:
            behavior = create_pf_behavior(args, data_i, beh_data)
            if args.with_explain:
                print(f"- new behavior {behavior.clause}")
            pf_behaviors.append(behavior)
    return pf_behaviors
