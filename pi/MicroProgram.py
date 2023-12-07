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
        exists_input = torch.tensor([x[i, i] > 0.8 for i in range(x.size(0))])
        exist_res = torch.prod(self.mask * exists_input)
        return exist_res.bool()

    def forward(self, x):
        # game Getout: tensor with size 1 * 4 * 6
        satisfies = True
        for i in range(len(self.pred_funcs)):
            data_A = x[self.obj_codes[i][0], self.prop_codes[i]]
            data_B = x[self.obj_codes[i][1], self.prop_codes[i]]
            func_satisfy = self.pred_funcs[i](data_A, data_B, self.obj_codes[i], batch_data=False)
            satisfies *= func_satisfy

            exist_satisfy = self.check_exists(x)
            satisfies *= exist_satisfy
        if satisfies:
            return self.action
        else:
            return None
