# Created by jing at 23.12.23
import torch


def get_param_range(min, max, unit):
    length = (max - min) // unit
    if length == 0:
        return torch.zeros(1)

    space = torch.zeros(int(length))
    for v_i in range(len(space)):
        space[v_i] = min + unit * v_i
    return space
