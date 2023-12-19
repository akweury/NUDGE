# Created by jing at 04.12.23


import torch
import torch.nn as nn


class MicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, action, mask, obj_codes, prop_codes, pred_funcs):
        super().__init__()
        self.action = action
        self.mask = mask
        self.obj_codes = obj_codes
        self.prop_codes = prop_codes
        self.pred_funcs = pred_funcs
        assert len(self.prop_codes) == 1

    def check_exists(self, x):
        if x.ndim != 3:
            raise ValueError

        mask_batches = self.mask.unsqueeze(0)

        state_exists = torch.ones(size=(x.size(0), x.size(1)), dtype=torch.bool)
        for i in range(x.size(1)):
            state_exists[:, i] *= (x[:, i, i] > 0.8)

        mask_batches = torch.repeat_interleave(mask_batches, x.size(0), dim=0)
        exist_res = torch.prod(mask_batches * state_exists, dim=1)
        return exist_res.bool()

    def forward(self, x, avg_data=False):
        # game Getout: tensor with size batch_size * 4 * 6
        satisfies = torch.ones(x.size(0), dtype=torch.bool)

        for i in range(len(self.pred_funcs)):
            data_A = x[:, self.obj_codes[i][0], self.prop_codes[i]]
            data_B = x[:, self.obj_codes[i][1], self.prop_codes[i]]
            func_satisfy = self.pred_funcs[i](data_A, data_B, self.obj_codes[i], batch_data=True, avg_data=avg_data)
            satisfies *= func_satisfy

            exist_satisfy = self.check_exists(x)
            satisfies *= exist_satisfy

        # return action probs
        action_probs = torch.zeros(x.size(0), len(self.action))
        action_probs[satisfies] += self.action

        return action_probs
