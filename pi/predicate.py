# Created by jing at 28.11.23
import torch


def ge(t1, t2, sr, avg_data=True, batch_data=True):
    if t1.sum() == 0 or t2.sum() == 0:
        return False

    # if t1.size(0)==0:
    #     return False

    # If all of A greater than B
    if avg_data:
        th = 1e-1
        var, mean = torch.var_mean(torch.ge(t1, t2).float())

        satisfy = False
        if var < th and (1 - mean) < th:
            satisfy = True
    else:
        satisfy = torch.ge(t1, t2).float().bool()
    return satisfy


def similar(t1, t2, sr, batch_data=True, avg_data=True):
    # repeat situation
    if sr[1] < sr[0] or t1.sum() == 0 or t2.sum() == 0:
        return False

    th = 0.5
    if avg_data:
        # If x distance between A and B is similar for all
        var, mean = torch.var_mean(torch.abs(torch.sub(t1, t2)))

        satisfy = False
        if torch.abs(var / mean) < th:
            satisfy = True
    else:
        dist = torch.abs(torch.sub(t1, t2))
        t1_t2 = torch.cat((t1.unsqueeze(1), t2.unsqueeze(1)), 1)
        satisfy = torch.abs(dist / t1_t2.min(dim=1)[0]) < th
    return satisfy


preds = [ge, similar]
pred_dict = {"greater_or_equal_than": ge,
             "as_similar_as": similar}
