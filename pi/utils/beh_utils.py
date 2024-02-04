# Created by jing at 31.01.24
import torch

from pi.utils.Behavior import Behavior
from pi.utils.Fact import ProbFact, VarianceFact
from pi import predicate
from pi import pi_lang
from pi.neural import nn_model
from pi.utils import draw_utils


def create_positive_behaviors(args, pos_beh_data):
    # create positive behaviors
    path_behaviors = []

    for beh_i, beh in enumerate(pos_beh_data):
        beh_facts = []
        for fact in beh["facts"]:
            beh_facts.append(ProbFact([predicate.GT()], fact["mask"], fact["objs"], fact["props"]))
        reward = beh["expected_reward"]
        passed_state_num = beh["passed_state_num"]
        test_passed_state_num = beh["test_passed_state_num"]
        failed_state_num = beh["failed_state_num"]
        test_failed_state_num = beh["test_failed_state_num"]
        action = beh["action"]
        neg_beh = False
        behavior = Behavior(neg_beh, beh_facts, action, reward, passed_state_num, test_passed_state_num,
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
        print(f"{beh.clause}, "
              f"+: {(beh.passed_state_num / beh.test_passed_state_num):.2f}, "
              f"-: {(beh.failed_state_num / beh.test_failed_state_num):.2f}. ")
    return path_behaviors


def create_negative_behaviors(args, def_beh_data):
    # create defense behaviors
    defense_behaviors = []
    for beh_i, beh in enumerate(def_beh_data):
        dists = torch.tensor(beh["dists"], dtype=torch.float32)
        dists_pos = torch.tensor(beh["dists_pos"], dtype=torch.float32)
        expected_reward = beh["rewards"]
        obj_combs = beh["obj_combs"]
        prop_combs = beh["prop_combs"]
        action_type = beh["action_type"]
        mask = beh["masks"]
        dists_var, dists_mean = torch.var_mean(dists)

        print(f'- (Behavior data) objs: {obj_combs}, props: {prop_combs}, action: {action_type}, mask: {mask}')
        draw_utils.plot_scatter([dists, dists_pos], ['positive', 'negative'],
                                f'{beh_i}_behavior_scatter_action_{action_type}',
                                args.output_folder, log_x=True)
        # generate enough data
        if len(dists_pos) > len(dists):
            generated_points = nn_model.generate_data(dists, gen_num=len(dists_pos) - len(dists))
            dists = torch.cat((dists, generated_points), dim=0)

        # prepare training data
        X = torch.cat((dists, dists_pos), dim=0)
        y = torch.zeros(len(X), 2)

        pos_indices = torch.cat(
            (torch.ones(len(dists), dtype=torch.bool), torch.zeros(len(dists_pos), dtype=torch.bool)), dim=0)
        y[pos_indices, 1] = 1
        y[~pos_indices, 0] = 1

        # fit a classifier using neural network
        num_epochs = 5000
        model = nn_model.fit_classifier(x_tensor=X, y_tensor=y, num_epochs=num_epochs)
        # plot decision boundary
        draw_utils.plot_decision_boundary(X, y, model,
                                          name=f"{beh_i}_behavior_db_action_{action_type}_var_{dists_var:.2f}_mean_{dists_mean:.2f}_ep_{num_epochs}",
                                          log_x=True,
                                          path=args.output_folder)
        # create predicate
        pred = [predicate.Dist(args, dists_var.tolist(), dists_mean.tolist(), model)]
        beh_fact = VarianceFact(dists, mask, obj_combs, prop_combs, pred)
        neg_beh = True

        behavior = Behavior(neg_beh, [beh_fact], action_type, expected_reward, len(dists), len(dists), 0, 0)
        behavior.clause = pi_lang.behavior2clause(args, behavior)
        print(f"{behavior.clause} "
              f"+: {(behavior.passed_state_num / ((behavior.test_passed_state_num) + 1e-20)):.2f}, "
              f"-: {(behavior.failed_state_num / (behavior.test_failed_state_num + 1e-20)):.2f}. ")
        defense_behaviors.append(behavior)
    return defense_behaviors
