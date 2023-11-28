# Created by jing at 28.11.23
import torch


def ge(t1, t2, sr):
    if t1.size(0) == 0 or t1.sum() == 0 or t2.sum() == 0:
        return False

    # If all of A greater than B
    th = 1e-1
    var, mean = torch.var_mean(torch.ge(t1, t2).float())

    satisfy = False
    if var < th and (1 - mean) < th:
        satisfy = True

    return satisfy


def similar(t1, t2, sr):
    # repeat situation
    if sr[1] < sr[0] or t1.size(0) == 0 or t1.sum() == 0 or t2.sum() == 0:
        return False

    th = 0.5
    # If x distance between A and B is similar for all
    var, mean = torch.var_mean(torch.abs(torch.sub(t1, t2)))

    # predicate similar(player_x, key_x)
    print(f'var/mean: {var / mean}')
    satisfy = False
    if torch.abs(var / mean) < th:
        satisfy = True

    return satisfy


preds = [ge, similar]
