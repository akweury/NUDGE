# Created by shaji at 26/01/2024
import torch

from src import config
from pi.utils import smp_utils


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

    def falsify_pred_params(self, preds, x, game_info):
        pred_tensor = self.fact['pred_tensors']
        prop = self.fact["props"]
        obj_type_combs = smp_utils.get_obj_type_combs(game_info, self.fact)

        satisafaction = []
        params = []
        for i in range(len(preds)):
            pred_satisfaction = []
            pred_params = []
            for obj_comb in obj_type_combs:
                data_A = x[:, obj_comb[0], prop].reshape(-1)
                data_B = x[:, obj_comb[1], prop].reshape(-1)
                satisfy, param = preds[i].eval(data_A, data_B)
                pred_satisfaction.append(satisfy)
                pred_params.append(param)
            satisafaction.extend(pred_satisfaction)
            params.extend(pred_params)

    def validify_pred_params(self, preds, x, game_info):
        pred_tensor = self.fact['pred_tensors']
        prop = self.fact["props"]
        obj_type_combs = smp_utils.get_obj_type_combs(game_info, self.fact)

        satisafaction = []
        params = []
        for i in range(len(preds)):
            pred_satisfaction = []
            pred_params = []
            for obj_comb in obj_type_combs:
                data_A = x[:, obj_comb[0], prop].reshape(-1)
                data_B = x[:, obj_comb[1], prop].reshape(-1)
                satisfy, param = preds[i].eval(data_A, data_B)
                pred_satisfaction.append(satisfy)
                pred_params.append(param)
            satisafaction.append(pred_satisfaction)
            params.append(pred_params)
        print('test')

    def eval_behavior(self, x, game_info):
        for fact in self.fact:
            type_0_index = fact["objs"][0]
            type_1_index = fact["objs"][1]
            prop = fact["props"]
            _, obj_0_indices, _ = game_info[type_0_index]
            _, obj_1_indices, _ = game_info[type_1_index]
            obj_combs = smp_utils.enumerate_two_combs(obj_0_indices, obj_1_indices)

            # check if current state has the same mask as the behavior
            fact_mask_tensor = smp_utils.mask_name_to_tensor(fact["mask"], config.mask_splitter)
            fact_mask_tensor = torch.repeat_interleave(fact_mask_tensor.unsqueeze(0), len(x), 0)
            mask_satisfaction = (fact_mask_tensor == smp_utils.mask_tensors_from_states(x, game_info)).prod(
                dim=-1).bool().reshape(-1)

            if not mask_satisfaction:
                return False

            # pred is true if any comb is true (or)
            fact_satisfaction = False
            for obj_comb in obj_combs:
                data_A = x[:, obj_comb[0], prop].reshape(-1)
                data_B = x[:, obj_comb[1], prop].reshape(-1)

                # behavior is true if all pred is true (and)
                state_pred_satisfaction = True
                for i in range(len(fact['preds'])):
                    if fact["pred_tensors"][i]:
                        pred_satisfaction = fact['preds'][i].eval(data_A, data_B)
                        state_pred_satisfaction *= pred_satisfaction
                # print(f"state preds: {state_pred_satisfaction}")
                fact_satisfaction += state_pred_satisfaction

            if not fact_satisfaction:
                return False

        return True
