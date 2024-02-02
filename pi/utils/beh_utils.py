# Created by jing at 31.01.24
import torch

from pi.utils.Behavior import Behavior
from pi.utils.Fact import ProbFact, VarianceFact
from pi import predicate
from pi import pi_lang


def create_positive_behaviors(args, pos_beh_data):
    # create positive behaviors
    path_behaviors = []
    for action_i in range(len(pos_beh_data)):
        for beh in pos_beh_data[action_i]:
            beh_facts = []
            for fact in beh["facts"]:
                beh_facts.append(
                    ProbFact([predicate.GT()], fact["mask"], fact["objs"], fact["props"][0], fact["delta"]))
            reward = beh["expected_reward"]
            passed_state_num = beh["passed_state_num"]
            test_passed_state_num = beh["test_passed_state_num"]
            failed_state_num = beh["failed_state_num"]
            test_failed_state_num = beh["test_failed_state_num"]
            neg_beh = False
            behavior = Behavior(neg_beh, beh_facts, action_i, reward, passed_state_num, test_passed_state_num,
                                failed_state_num,
                                test_failed_state_num)
            behavior.clause = pi_lang.behavior2clause(args, behavior)
            path_behaviors.append(behavior)

    behaviors = []

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
        print(f"{beh.clause}, "
              f"+: {(beh.passed_state_num / beh.test_passed_state_num):.2f}, "
              f"-: {(beh.failed_state_num / beh.test_failed_state_num):.2f}. ")
    return path_behaviors


def create_negative_behaviors(args, def_beh_data):
    # create defense behaviors
    defense_behaviors = []
    for beh in def_beh_data:
        dists = beh["dists"]
        dists_pos = beh["dists_pos"]
        expected_reward = beh["rewards"]
        obj_combs = beh["obj_combs"]
        prop_combs = beh["prop_combs"]
        action_type = beh["action_type"]
        mask = beh["masks"]
        delta = beh["delta"]
        dists_var, dists_mean = torch.var_mean(torch.tensor(dists))
        pred = [predicate.Dist(dists,dists_pos, dists_var.tolist(), dists_mean.tolist())]
        beh_fact = VarianceFact(dists, mask, obj_combs, prop_combs, pred, delta)
        neg_beh = True

        behavior = Behavior(neg_beh, [beh_fact], action_type, expected_reward, len(dists), len(dists), 0, 0)
        behavior.clause = pi_lang.behavior2clause(args, behavior)
        print(f"{behavior.clause} "
              f"+: {(behavior.passed_state_num / ((behavior.test_passed_state_num) + 1e-20)):.2f}, "
              f"-: {(behavior.failed_state_num / (behavior.test_failed_state_num + 1e-20)):.2f}. ")
        defense_behaviors.append(behavior)
    return defense_behaviors
