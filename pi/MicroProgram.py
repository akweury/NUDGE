# Created by jing at 04.12.23


import torch
import torch.nn as nn


class MicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, action, mask, obj_codes, prop_codes, pred_funcs, p_bound, p_space):
        super().__init__()
        self.action = action
        self.mask = mask
        self.obj_codes = obj_codes
        self.prop_codes = prop_codes
        self.pred_funcs = pred_funcs
        self.p_bound = p_bound
        self.p_space = p_space
        assert len(self.prop_codes) == 1

    def check_exists(self, x):
        if x.ndim != 3:
            raise ValueError

        mask_batches = self.mask.unsqueeze(0)

        state_exists = torch.ones(size=(x.size(0), x.size(1)), dtype=torch.bool)
        for i in range(x.size(1)):
            state_exists[:, i] *= (x[:, i, i] > 0.8)
        mask_batches = torch.repeat_interleave(mask_batches, x.size(0), dim=0)
        exist_res = torch.prod(mask_batches == state_exists, dim=1)
        return exist_res.bool()

    def forward(self, x, avg_data=False):
        # game Getout: tensor with size batch_size * 4 * 6
        satisfies = torch.ones(x.size(0), dtype=torch.bool)
        # if use_given_parameters:
        #     given_parameters = self.p_bound
        # else:
        #     given_parameters = None

        data_A = x[:, self.obj_codes[0], self.prop_codes].reshape(-1)
        data_B = x[:, self.obj_codes[1], self.prop_codes].reshape(-1)
        func_satisfy, p_values = self.pred_funcs.eval(data_A, data_B, self.p_space)

        satisfies *= func_satisfy

        exist_satisfy = self.check_exists(x)
        satisfies *= exist_satisfy

        # return action probs
        action_probs = torch.zeros(x.size(0), len(self.action))
        action_probs[satisfies] += self.action

        action_probs[satisfies] = action_probs[satisfies] / (action_probs[satisfies] + 1e-20)

        return action_probs, p_values
