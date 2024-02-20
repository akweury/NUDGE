# Created by shaji at 08/02/2024
import torch
import math
import numpy as np
from collections import Counter


def calculate_direction(points, reference_point):
    directions = []
    for r_i in range(points.shape[0]):
        x_ref, y_ref = reference_point[0].squeeze()

        x, y = points[r_i].squeeze()
        delta_x = x - x_ref
        delta_y = y_ref- y

        angle_radians = torch.atan2(delta_y, delta_x)
        angle_degrees = math.degrees(angle_radians)

        directions.append(angle_degrees)

    return directions


def degrees_to_direction(degrees):
    # Ensure degrees are within the range [0, 360)
    degrees = degrees % 360

    # Define directional sectors
    sectors = ['right', 'up-right', 'up', 'up-left', 'left', 'down-left', 'down', 'down-right']

    # Determine the index of the closest sector
    index = (torch.round(degrees / 45) % 8).to(torch.int)

    # Return the corresponding direction
    return sectors[index]


def range_to_direction(given_range):
    mid_value = (given_range[0] + (given_range[1] - given_range[0]) * 0.5) * 360
    dir_degree = closest_multiple_of_45(mid_value)
    return dir_degree


def get_frequnst_value(data):
    unique_values, counts = np.unique(data.reshape(-1), return_counts=True)
    most_frequency_value = unique_values[np.argmax(counts)]
    return most_frequency_value


def get_90_percent_range_2d(data):
    """
    Get the range that covers 90% of the data for each dimension using PyTorch's percentile.
    Assumes the data is a 2D PyTorch tensor of shape (n_samples, 2).
    Returns two tuples (x_range, y_range) representing the ranges for each dimension.
    """
    # Calculate the 5th and 95th percentiles for each dimension

    ranges = np.zeros((data.shape[-1], 2))
    for i in range(data.shape[-1]):
        ranges[i] = np.percentile(data[:, i], [2, 98])
    return ranges


def one_step_move(data, direction, distance):
    """
    Move a list of 2D points along a given direction by a specified distance.

    Parameters:
    - points: List of 2D points [(x1, y1), (x2, y2), ...]
    - direction: Tuple representing the direction vector (dx, dy)
    - distance: Distance to move points along the direction

    Returns:
    - List of moved points
    """
    if direction is None or abs(direction) > 1:
        direction = 0
        distance = [0, 0]
    direction_rad = math.radians(direction * 180)
    dx = math.cos(direction_rad)
    dy = math.sin(direction_rad)
    direction_vec = torch.tensor([dx, dy]).to(data.device)
    direction_unit_vector = direction_vec / torch.norm(direction_vec)
    new_points = data + direction_unit_vector * torch.tensor(distance).to(data.device)

    return new_points


def dist_a_and_b_closest(data_A, data_B):
    closest_index = torch.zeros(data_A.shape[0])
    if len(data_B.size()) == 3:
        diff_abs = torch.sub(torch.repeat_interleave(data_A, data_B.size(1), 1), data_B)
        dist_all = torch.zeros(data_B.size(0), data_B.size(1))

        for d_i in range(data_B.size(1)):
            dist_all[:, d_i] = torch.norm(diff_abs[:, d_i, :], dim=1)

        _, closest_index = torch.abs(dist_all).min(dim=1)
        dist = torch.zeros(data_B.size(0), data_B.size(2))
        for i in range(closest_index.size(0)):
            dist[i] = torch.abs(diff_abs[i, closest_index[i], :])
    else:
        dist = data_A - data_B

    return dist, closest_index


def dist_a_and_b(data_A, data_B):
    diff_abs = torch.abs(torch.sub(data_A, data_B))

    return diff_abs


def cart2pol(x, y):
    rho = torch.sqrt(x ** 2 + y ** 2)
    phi = torch.atan2(y, x)
    phi = torch.rad2deg(phi)
    return (rho, phi)


# def dir_a_and_b(data_A, data_B):
#     dir_vec = torch.sub(data_B, data_A)
#     dir_vec[1] = -dir_vec[1]
#     rho, phi = cart2pol(dir_vec[0], dir_vec[1])
#     assert (torch.abs(phi) <= 180).prod() == True
#     dir = phi / 180
#
#     return dir

def closest_quarter(dir_value):
    rounded_dir = torch.round(dir_value /0.25 ) * 0.25
    return rounded_dir
def closest_multiple_of_45(degrees):
    # Ensure the input degree is within the range [0, 360]
    degrees = torch.tensor(degrees)
    degrees = degrees % 360

    # Calculate the remainder when dividing by 45
    remainder = degrees % 45

    # Determine the closest multiple of 45
    closest_multiple = degrees - remainder

    # Check if rounding up is closer than rounding down

    closest_multiple[remainder > 22.5] += 45
    closest_multiple[closest_multiple <= 180] /= 180
    closest_multiple[closest_multiple > 180] = -(360 - closest_multiple[closest_multiple > 180]) / 180
    return closest_multiple


def dir_ab_batch(data_A, data_B, indices):
    directions = []
    for d_i in range(data_A.shape[0]):
        index = indices[d_i]
        a = data_A[d_i]
        b = data_B[d_i][index:index + 1]
        dir = dir_a_and_b(a,b).tolist()
        # dir = dir_a_and_b_with_alignment(a, b).tolist()
        directions.append(dir)
    directions = torch.tensor(directions)
    return directions

def dir_a_and_b(data_A, data_B):
    directions_in_degree = np.array(calculate_direction(data_B, data_A))

    directions_in_degree[directions_in_degree <= 180] /= 180
    directions_in_degree[directions_in_degree > 180] = -(360 - directions_in_degree[directions_in_degree > 180]) / 180
    # directions_aligned = closest_multiple_of_45(directions_in_degree).unsqueeze(1)

    return directions_in_degree

def dir_a_and_b_with_alignment(data_A, data_B):
    directions_in_degree = calculate_direction(data_B, data_A)
    directions_aligned = closest_multiple_of_45(directions_in_degree).unsqueeze(1)

    return directions_aligned


def action_to_deg(action_name):
    if action_name == "noop":
        dir = 100
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


def pol2dir_name(dir_mean):
    if -0.05 <= dir_mean < 0.05:
        dir_name = "right"
    elif 0.05 <= dir_mean < 0.45:
        dir_name = "upright"
    elif 0.45 <= dir_mean < 0.55:
        dir_name = "up"
    elif 0.55 <= dir_mean < 0.95:
        dir_name = "upleft"
    elif 0.95 <= dir_mean <= 1 or -1 <= dir_mean <= -0.95:
        dir_name = "left"
    elif -0.95 <= dir_mean < -0.55:
        dir_name = "downleft"
    elif -0.55 <= dir_mean < -0.45:
        dir_name = "down"
    elif -0.45 <= dir_mean < -0.05:
        dir_name = "downright"
    else:
        raise ValueError

    return dir_name


