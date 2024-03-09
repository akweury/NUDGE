# Created by shaji at 08/02/2024
import torch
import math
import numpy as np
import torch.nn.functional as F
from itertools import chain, combinations

from collections import Counter


def calculate_direction(points, reference_point):
    x_ref, y_ref = reference_point[0, :, 0].squeeze(), reference_point[0, :, 1].squeeze()
    x, y = points[:, :, 0].squeeze(), points[:, :, 1].squeeze()
    delta_x = x - x_ref
    delta_y = y_ref - y

    angle_radians = torch.atan2(delta_y, delta_x)
    angle_degrees = torch.rad2deg(angle_radians)

    return angle_degrees


def calculate_direction_o2o(points, reference_point):
    directions = []
    for r_i in range(points.shape[1]):
        x_ref, y_ref = reference_point[:, 0, 0], reference_point[:, 0, 1]

        x, y = points[:, r_i, 0], points[:, r_i, 1]
        delta_x = x - x_ref
        delta_y = y_ref - y

        angle_radians = torch.atan2(delta_y, delta_x)
        angle_degrees = torch.rad2deg(angle_radians)

        directions.append(angle_degrees.tolist())

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


def one_step_move_o2o(data, direction, distance):
    """
    Move a list of 2D points along a given direction by a specified distance.

    Parameters:
    - points: List of 2D points [(x1, y1), (x2, y2), ...]
    - direction: Tuple representing the direction vector (dx, dy)
    - distance: Distance to move points along the direction

    Returns:
    - List of moved points
    """

    direction_rad = torch.deg2rad(direction * 180)
    dx = torch.cos(direction_rad)
    dy = torch.sin(direction_rad)
    direction_vec = torch.cat((dx.unsqueeze(0), dy.unsqueeze(0)), dim=0).to(data.device)

    direction_unit_vector = (direction_vec / torch.norm(direction_vec, dim=0)).permute(1, 0)
    direction_unit_vector[direction > 1] = torch.zeros(2).to(data.device)
    direction_unit_vector = torch.repeat_interleave(direction_unit_vector.unsqueeze(1), data.shape[1], dim=1)
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


def all_subsets(input_list):
    # Convert the input list to a tuple for use in combinations
    input_tuple = tuple(input_list)

    # Generate all possible subsets using combinations
    subsets = chain.from_iterable(combinations(input_tuple, r) for r in range(1, len(input_tuple) + 1))

    # Convert the subsets from tuples back to lists
    subsets = [list(subset) for subset in subsets]
    return subsets


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
def closest_one_percent(dist_value, unit=0.001):
    rounded_dist = torch.round(dist_value / unit) * unit
    return rounded_dist


def closest_quarter(dir_value):
    rounded_dir = torch.round(dir_value / 0.25) * 0.25
    return rounded_dir


def closest_multiple_of_45(degrees):
    # Ensure the input degree is within the range [0, 360]

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
        dir = dir_a_and_b(a, b).tolist()
        # dir = dir_a_and_b_with_alignment(a, b).tolist()
        directions.append(dir)
    directions = torch.tensor(directions)
    return directions


def dir_ab_any(data_A, data_B):
    directions = torch.zeros(data_B.shape[0], data_B.shape[1]).to(data_A.device)
    for d_i in range(data_A.shape[0]):
        a = data_A[d_i]
        b = data_B[d_i]
        dir = dir_a_and_b_tensor(a, b)
        directions[d_i] = dir
    return directions


def dir_a_and_b_tensor(data_A, data_B):
    directions_in_degree = calculate_direction(data_B, data_A)

    directions_in_degree[directions_in_degree <= 180] /= 180
    directions_in_degree[directions_in_degree > 180] = -(360 - directions_in_degree[directions_in_degree > 180]) / 180
    # directions_aligned = closest_multiple_of_45(directions_in_degree).unsqueeze(1)

    return directions_in_degree


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


def dir_a_and_b_with_alignment_o2o(data_A, data_B):
    directions_in_degree = calculate_direction_o2o(data_B, data_A)
    directions_aligned = closest_multiple_of_45(directions_in_degree).unsqueeze(1)

    return directions_aligned


def action_to_deg(action_name):
    if action_name == "noop" or action_name == "fire" or action_name == "jump":
        dir = 100
    elif action_name == "up" or action_name == "upfire":
        dir = 90 / 180
    elif action_name == "right" or action_name == "rightfire":
        dir = 0 / 180
    elif action_name == "left" or action_name == "leftfire":
        dir = 180 / 180
    elif action_name == "down" or action_name == "downfire":
        dir = -90 / 180
    elif action_name == "upright" or action_name == "uprightfire":
        dir = 45 / 180
    elif action_name == "upleft" or action_name == "upleftfire":
        dir = 135 / 180
    elif action_name == "downright" or action_name == "downrightfire":
        dir = -45 / 180
    elif action_name == "downleft" or action_name == "downleftfire":
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


def calculate_acceleration_2d(positions_x, positions_y):
    times = [0, 1, 2]
    # Ensure we have enough data points
    if len(positions_x) == len(positions_y) == len(times) == 3:
        # Calculate changes in velocity in each dimension
        delta_v_x = (positions_x[2] - positions_x[0]) / (times[2] - times[0])
        delta_v_y = (positions_y[2] - positions_y[0]) / (times[2] - times[0])

        # Calculate accelerations in each dimension
        acceleration_x = delta_v_x / (times[2] - times[0])
        acceleration_y = delta_v_y / (times[2] - times[0])

        return torch.cat((acceleration_x.unsqueeze(0), acceleration_y.unsqueeze(0)))
    else:
        print("Insufficient data points. Need positions_x, positions_y.")


def calculate_velocity_2d(positions_x, positions_y):
    times = [0, 1]
    # Ensure we have enough data points
    if len(positions_x) == len(positions_y) == len(times) == 2:
        # Calculate changes in velocity in each dimension
        delta_v_x = (positions_x[1] - positions_x[0]) / (times[1] - times[0])
        delta_v_y = (positions_y[1] - positions_y[0]) / (times[1] - times[0])
        return torch.cat((delta_v_x.unsqueeze(0), delta_v_y.unsqueeze(0)))
    else:
        print("Insufficient data points. Need positions_x, positions_y.")


def discounted_rewards(rewards, gamma=0.2):
    discounted = []
    running_add = 0
    for r in reversed(rewards):
        running_add = running_add * gamma + r
        discounted.insert(0, running_add)
    return torch.tensor(discounted)


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def smooth_filter(data, window_size=5):
    # Set the window size for the moving average
    # Define the moving average filter
    kernel = torch.ones(window_size) / window_size
    actions_smooth = F.conv1d(data.view(1, 1, -1),
                              kernel.view(1, 1, -1),
                              padding=(window_size - 1) // 2)[0, 0, :]
    return actions_smooth


def get_velo_dir(velo):
    velo_dir = torch.atan2(velo[:, :, 1], velo[:, :, 0])
    velo_dir = torch.rad2deg(velo_dir)
    velo_dir = (torch.round(velo_dir / 45) % 8)
    velo_dir = normalize(velo_dir)
    return velo_dir


# Define a function to remove outliers based on the interquartile range (IQR)
def remove_outliers_iqr(data, factor=1.5):
    # Calculate the first and third quartiles (Q1 and Q3)
    q1 = torch.quantile(data, 0.25)
    q3 = torch.quantile(data, 0.75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    # Remove outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return filtered_data


def common_elements(lists):
    # Use the set intersection to find common elements
    values_0 = set(lists[0])
    for one_list in lists:
        values_0 = values_0.intersection(one_list)

    return list(values_0)


def non_sublists(lists):
    result = []
    indices = []
    for l_i, lst1 in enumerate(lists):
        is_sublist = False
        for lst2 in lists:
            if lst1 != lst2 and all(item in lst2 for item in lst1):
                is_sublist = True
                break
        if not is_sublist:
            result.append(lst1)
            indices.append(l_i)
    return result, indices


def is_sublist(sublist, mainlist):
    return all(item in mainlist for item in sublist)


def indices_of_minimum(tensor):
    min_value = tensor.min().item()  # Find the minimum value
    min_indices = (tensor == min_value).nonzero()  # Find indices where the value equals the minimum

    return min_indices


def indices_of_maximum(tensor):
    max_value = tensor.max().item()  # Find the maximum value
    max_indices = (tensor == max_value).nonzero()  # Find indices where the value equals the maximum

    return max_indices
