# Created by jing at 31.01.24
import torch
import itertools
from src import config


class Behavior():
    """ generate one micro-program
    """

    def __init__(self, neg_beh, facts, action, reward, passed_state_num, test_passed_state_num, failed_state_num,
                 test_failed_state_num):
        super().__init__()
        self.fact = facts
        self.action = action
        self.reward = reward
        self.passed_state_num = passed_state_num
        self.test_passed_state_num = test_passed_state_num
        self.failed_state_num = failed_state_num
        self.test_failed_state_num = test_failed_state_num
        self.neg_beh = neg_beh
        self.clause = None

    def mask_tensors_from_states(self, states, game_info):
        mask_tensors = torch.zeros((len(states), len(game_info)), dtype=torch.bool)
        for i in range(len(game_info)):
            name, obj_indices, prop_index = game_info[i]
            obj_exist_counter = states[:, obj_indices, prop_index].sum()
            mask_tensors[:, i] = obj_exist_counter > 0
        mask_tensors = mask_tensors.bool()
        return mask_tensors

    def eval_behavior(self, x, game_info):
        prediction = torch.zeros(len(self.fact))
        for f_i, fact in enumerate(self.fact):
            type_0_index = fact.obj_comb[0]
            type_1_index = fact.obj_comb[1]
            prop = fact.prop_comb
            _, obj_0_indices, _ = game_info[type_0_index]
            _, obj_1_indices, _ = game_info[type_1_index]
            obj_combs = list(itertools.product(obj_0_indices, obj_1_indices))
            # check if current state has the same mask as the behavior
            fact_mask_tensor = torch.repeat_interleave(torch.tensor(fact.mask).unsqueeze(0), len(x), 0)
            mask_satisfaction = (fact_mask_tensor == self.mask_tensors_from_states(x, game_info)).prod(
                dim=-1).bool().reshape(-1)
            if not mask_satisfaction:
                return prediction
            # pred is true if any comb is true (or)
            fact_satisfaction = False
            for obj_comb in obj_combs:
                data_A = x[:, obj_comb[0], prop].reshape(-1)
                data_B = x[:, obj_comb[1], prop].reshape(-1)
                # behavior is true if all pred is true (and)
                prediction[f_i:f_i + 1] += fact.preds[0].eval(data_A, data_B)

            prediction[f_i] = prediction / (len(obj_combs) + 1e-20)

        prediction = prediction.mean()
        if prediction > 1:
            print("")
        return prediction
