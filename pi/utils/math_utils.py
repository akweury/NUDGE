# Created by shaji at 08/02/2024
import torch
import math


def calculate_direction(points, reference_points):
    directions = []
    for r_i, reference_point in enumerate(reference_points.squeeze()):
        directions_ri = []
        x_ref, y_ref = reference_point
        for point in points[r_i].squeeze():
            x, y = point
            delta_x = x - x_ref
            delta_y = y - y_ref

            angle_radians = torch.atan2(delta_y, delta_x)
            angle_degrees = math.degrees(angle_radians)

            directions_ri.append(angle_degrees)
        directions.append(directions_ri)
    directions = torch.tensor(directions) / 180

    return directions


def dist_a_and_b_closest(data_A, data_B, reference_dir, deg_th=20 / 180):
    closest_index = torch.zeros(data_A.shape[0])
    if len(data_B.size()) == 3:

        diff_abs = torch.abs(torch.sub(torch.repeat_interleave(data_A, data_B.size(1), 1), data_B))
        dist_all = torch.zeros(data_B.size(0), data_B.size(1))

        for d_i in range(data_B.size(1)):
            dist_all[:, d_i] = torch.norm(diff_abs[:, d_i, :], dim=1)
        dir_all = calculate_direction(data_B, data_A)
        if reference_dir is not None:
            angel_err = dir_all - reference_dir
            mask = torch.abs(angel_err) > deg_th
            dist_all[mask] += angel_err[mask]
            closest_dist, closest_index = dist_all.min(dim=1)
        else:
            closest_dist, closest_index = dist_all.min(dim=1)
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
        dir_vec = torch.sub(data_B_closest, data_A[i, 0])
        dir_vec[1] = -dir_vec[1]
        rho, phi = cart2pol(dir_vec[0], dir_vec[1])
        dir[i] = phi
    assert (torch.abs(dir) <= 180).prod() == True
    dir = dir / 180

    return dir


def action_to_deg(action_name):
    if action_name == "noop":
        dir = None
    elif action_name == "up":
        dir = 90 / 180
    elif action_name == "right":
        dir = 0 / 180
    elif action_name == "left":
        dir = 180 / 180
    elif action_name == "down":
        dir = -90 / 180
    elif action_name == "upright":
        dir = 45 / 180
    elif action_name == "upleft":
        dir = 135 / 180
    elif action_name == "downright":
        dir = -45 / 180
    elif action_name == "downleft":
        dir = -135 / 180
    else:
        raise ValueError
    return dir
