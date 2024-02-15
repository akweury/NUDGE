# Created by shaji at 08/02/2024
import torch


def dist_a_and_b_closest(data_A, data_B):
    closest_index = torch.zeros(data_A.shape[0])
    if len(data_B.size()) == 3:
        diff_abs = torch.abs(torch.sub(torch.repeat_interleave(data_A, data_B.size(1), 1), data_B))
        dist_all = torch.zeros(data_B.size(0), data_B.size(1))
        for d_i in range(data_B.size(1)):
            dist_all[:, d_i] = torch.norm(diff_abs[:, d_i, :], dim=1)
        _, closest_index = dist_all.min(dim=1)
        dist = torch.zeros(data_B.size(0), data_B.size(2))
        for i in range(closest_index.size(0)):
            dist[i] = diff_abs[i, closest_index[i], :]

    else:
        dist = torch.abs(data_A - data_B)

    return dist, closest_index


def cart2pol(x, y):
    rho = torch.sqrt(x ** 2 + y ** 2)
    phi = torch.atan2(y, x)
    phi = torch.rad2deg(phi)
    return (rho, phi)


def dir_a_and_b_closest(data_A, data_B, b_indices):
    dir = torch.zeros(data_A.shape[0], data_A.shape[1])
    for i in range(data_B.shape[0]):
        data_B_closest = data_B[i, b_indices[i]]
        dir_vec = torch.sub(data_B_closest, data_A[i,0])
        dir_vec[1] = -dir_vec[1]
        rho, phi = cart2pol(dir_vec[0], dir_vec[1])
        dir[i] = phi
    return dir
