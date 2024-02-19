# Created by jing at 31.01.24
import torch
import itertools
from src import config


class Behavior():
    """ generate one micro-program
    """

    def __init__(self, beh_type, neg_beh, facts, action, reward, passed_state_num, test_passed_state_num, failed_state_num,
                 test_failed_state_num):
        super().__init__()
        self.beh_type=beh_type
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
            obj_indices = game_info[i]["indices"]
            obj_exist_counter = states[:, obj_indices, i].sum()
            mask_tensors[:, i] = obj_exist_counter > 0
        mask_tensors = mask_tensors.bool()
        return mask_tensors

    def eval_behavior(self, x, game_info):
        prediction = torch.zeros(len(self.fact), dtype=torch.bool)
        confidence = torch.zeros(len(self.fact))
        for f_i, fact in enumerate(self.fact):
            type_0_index = fact.obj_comb[0]
            type_1_index = fact.obj_comb[1]
            prop = fact.prop_comb
            obj_0_indices = game_info[type_0_index]["indices"]
            obj_1_indices = game_info[type_1_index]["indices"]
            obj_combs = torch.tensor(list(itertools.product(obj_0_indices, obj_1_indices)))
            # check if current state has the same mask as the behavior
            fact_mask_tensor = torch.repeat_interleave(torch.tensor(fact.mask).unsqueeze(0), len(x), 0)
            mask_satisfaction = (fact_mask_tensor == self.mask_tensors_from_states(x, game_info)).prod(
                dim=-1).bool().reshape(-1)
            if not mask_satisfaction:
                return prediction
            # pred is true if any comb is true (or)
            fact_satisfaction = False
            obj_a_indices = obj_combs[:, 0].unique()
            obj_b_indices = obj_combs[:, 1].unique()
            if len(obj_a_indices) == 1:
                data_A = x[:, obj_a_indices][:, :, prop]
                # try to find the closest obj B
                data_B = x[:, obj_b_indices][:, :, prop]
                # behavior is true if all pred is true (and)
                esti = fact.preds[0].eval(data_A, data_B, self.action, self.beh_type)
                if self.beh_type == "attack" and esti >10:
                    print(f"esti: {esti:.1f} {self.clause}")
                    a=True

                confidence[f_i:f_i + 1] = fact.preds[0].eval(data_A, data_B, self.action, self.beh_type)
            prediction[f_i] = prediction / (len(obj_combs) + 1e-20)
        return confidence
