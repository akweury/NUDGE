# Created by jing at 04.12.23


import torch
import torch.nn as nn


class MicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, obj_type_name, action, mask, type_codes, prop_codes, preds, p_spaces,
                 p_satisfication):
        super().__init__()
        self.obj_type_names = obj_type_name
        self.action = action
        self.mask = mask
        self.type_codes = type_codes
        self.prop_codes = prop_codes
        self.preds = preds
        self.p_spaces = p_spaces
        self.p_satisfication = p_satisfication
        self.obj_type_num = len(obj_type_name)

        assert len(self.prop_codes) == 1

    def check_exists(self, x, obj_type_dict):
        if x.ndim != 3:
            raise ValueError
        state_num = x.size(0)
        mask_batches = self.mask.unsqueeze(0)
        obj_type_exists = torch.ones(size=(state_num, self.obj_type_num), dtype=torch.bool)
        for obj_type_name, obj_type_indices in obj_type_dict.items():
            if obj_type_name in self.obj_type_names:
                obj_indices = [n_i for n_i, name in enumerate(self.obj_type_names) if name == obj_type_name]
                exist_objs = (x[:, obj_type_indices, obj_indices] > 0.8)
                exist_type = exist_objs.prod(dim=-1, keepdims=True).bool()
                obj_type_exists[:, obj_indices] *= exist_type

        mask_batches = torch.repeat_interleave(mask_batches, x.size(0), dim=0)
        exist_res = torch.prod(mask_batches == obj_type_exists, dim=1)
        return exist_res.bool()

    def forward(self, x, obj_type_indices, avg_data=False):
        # game Getout: tensor with size batch_size * 4 * 6
        satisfies = torch.zeros(x.size(0), dtype=torch.bool)
        # if use_given_parameters:
        #     given_parameters = self.p_bound
        # else:
        #     given_parameters = None
        type_1_name = self.obj_type_names[self.type_codes[0]]
        type_1_obj_codes = obj_type_indices[type_1_name]
        type_2_name = self.obj_type_names[self.type_codes[1]]
        type_2_obj_codes = obj_type_indices[type_2_name]

        if len(type_2_obj_codes) > 1 or len(type_1_obj_codes) > 1:
            print("WARNING:")
        # check predicates satisfaction
        for obj_1 in type_1_obj_codes:
            for obj_2 in type_2_obj_codes:
                data_A = x[:, obj_1, self.prop_codes].reshape(-1)
                data_B = x[:, obj_2, self.prop_codes].reshape(-1)

                obj_comb_satisfies = torch.ones(x.size(0), dtype=torch.bool)
                p_spaces = []
                for p_i, pred in enumerate(self.preds):
                    p_space = self.p_spaces[p_i]
                    p_satisfied = self.p_satisfication[p_i]
                    if not p_satisfied:
                        func_satisfy, p_values = torch.ones(data_A.size()).bool(), torch.zeros(size=data_A.size())
                    else:
                        func_satisfy, p_values = pred.eval(data_A, data_B, p_space)
                    p_spaces.append(p_values.unsqueeze(0))

                    # satisfy all
                    obj_comb_satisfies *= func_satisfy

                # satisfy any
                satisfies += obj_comb_satisfies

        # check mask satisfaction
        exist_satisfy = self.check_exists(x, obj_type_indices)

        # satisfy all
        satisfies *= exist_satisfy

        # return action probs
        action_probs = torch.zeros(x.size(0), len(self.action))
        action_probs[satisfies] += self.action
        action_probs[satisfies] = action_probs[satisfies] / (action_probs[satisfies] + 1e-20)

        return action_probs, p_spaces


class UngroundedMicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, obj_type_name, action, mask, type_codes, prop_codes, preds, p_spaces,
                 p_satisfication):
        super().__init__()
        self.obj_type_names = obj_type_name
        self.action = action
        self.mask = mask
        self.type_codes = type_codes
        self.prop_codes = prop_codes
        self.preds = preds
        self.p_spaces = p_spaces
        self.p_satisfication = p_satisfication
        self.obj_type_num = len(obj_type_name)

        assert len(self.prop_codes) == 1

    def check_exists(self, x, obj_type_dict):
        if x.ndim != 3:
            raise ValueError
        state_num = x.size(0)
        mask_batches = self.mask.unsqueeze(0)
        obj_type_exists = torch.ones(size=(state_num, self.obj_type_num), dtype=torch.bool)
        for obj_type_name, obj_type_indices in obj_type_dict.items():
            if obj_type_name in self.obj_type_names:
                obj_indices = [n_i for n_i, name in enumerate(self.obj_type_names) if name == obj_type_name]
                exist_objs = (x[:, obj_type_indices, obj_indices] > 0.8)
                exist_type = exist_objs.prod(dim=-1, keepdims=True).bool()
                obj_type_exists[:, obj_indices] *= exist_type

        mask_batches = torch.repeat_interleave(mask_batches, x.size(0), dim=0)
        exist_res = torch.prod(mask_batches == obj_type_exists, dim=1)
        return exist_res.bool()

    def forward(self, x, obj_type_indices, avg_data=False):
        # game Getout: tensor with size batch_size * 4 * 6
        satisfies = torch.zeros(x.size(0), dtype=torch.bool)
        # if use_given_parameters:
        #     given_parameters = self.p_bound
        # else:
        #     given_parameters = None
        type_1_name = self.obj_type_names[self.type_codes[0]]
        type_1_obj_codes = obj_type_indices[type_1_name]
        type_2_name = self.obj_type_names[self.type_codes[1]]
        type_2_obj_codes = obj_type_indices[type_2_name]

        if len(type_2_obj_codes) > 1 or len(type_1_obj_codes) > 1:
            print("WARNING:")
        # check predicates satisfaction
        for obj_1 in type_1_obj_codes:
            for obj_2 in type_2_obj_codes:
                data_A = x[:, obj_1, self.prop_codes].reshape(-1)
                data_B = x[:, obj_2, self.prop_codes].reshape(-1)

                obj_comb_satisfies = torch.ones(x.size(0), dtype=torch.bool)
                p_spaces = []
                for p_i, pred in enumerate(self.preds):
                    p_space = self.p_spaces[p_i]
                    p_satisfied = self.p_satisfication[p_i]
                    if not p_satisfied:
                        func_satisfy, p_values = torch.ones(data_A.size()).bool(), torch.zeros(size=data_A.size())
                    else:
                        func_satisfy, p_values = pred.eval(data_A, data_B, p_space)
                    p_spaces.append(p_values.unsqueeze(0))

                    # satisfy all
                    obj_comb_satisfies *= func_satisfy

                # satisfy any
                satisfies += obj_comb_satisfies

        # check mask satisfaction
        exist_satisfy = self.check_exists(x, obj_type_indices)

        # satisfy all
        satisfies *= exist_satisfy

        # return action probs
        action_probs = torch.zeros(x.size(0), len(self.action))
        action_probs[satisfies] += self.action
        action_probs[satisfies] = action_probs[satisfies] / (action_probs[satisfies] + 1e-20)

        return action_probs, p_spaces
