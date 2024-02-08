# Created by shaji at 08/02/2024
import torch


def dist_a_and_b_closest(data_A, data_B):
    if len(data_B.size()) == 3:
        try:
            diff_abs = torch.abs(torch.sub(torch.repeat_interleave(data_A, data_B.size(1), 1), data_B))
        except RuntimeError:
            print("")
        dist_all = torch.zeros(data_B.size(0), data_B.size(1))
        for d_i in range(data_B.size(1)):
            dist_all[:, d_i] = torch.norm(diff_abs[:, d_i, :], dim=1)
        _, closest_index = dist_all.min(dim=1)
        dist = torch.zeros(data_B.size(0), data_B.size(2))
        for i in range(closest_index.size(0)):
            dist[i] = diff_abs[i, closest_index[i],:]
    else:
        dist = torch.abs(data_A - data_B)

    return dist
