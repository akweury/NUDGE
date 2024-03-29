# Created by jing at 01.03.24
import torch
import numpy as np
from itertools import combinations
from pi.utils import file_utils, math_utils, draw_utils
from itertools import product
from sklearn.linear_model import LinearRegression


def discounted_rewards(rewards, gamma=0.2, alignment=None):
    discounted = []
    running_add = 0
    for r in reversed(rewards):
        running_add = running_add * gamma + r
        discounted.insert(0, running_add)
    discounted = torch.tensor(discounted)
    if alignment is not None:
        discounted = math_utils.closest_one_percent(discounted, alignment)
    return discounted


def get_state_velo(states):
    if len(states) == 1:
        velocity = torch.zeros(1, states.shape[1], 2)
    else:
        state_2 = torch.cat((states[:-1, :, -2:].unsqueeze(0), states[1:, :, -2:].unsqueeze(0)), dim=0)
        velocity = math_utils.calculate_velocity_2d(state_2[:, :, :, 0], state_2[:, :, :, 1]).permute(1, 2, 0)
        velocity = math_utils.closest_one_percent(velocity, 0.01)
        velocity = torch.cat((torch.zeros(1, velocity.shape[1], velocity.shape[2]).to(velocity.device), velocity),
                             dim=0)
    return velocity


def get_ab_dir(states, obj_a_index, obj_b_index):
    points_a = states[:, obj_a_index, -2:]
    points_b = states[:, obj_b_index, -2:]
    x_ref, y_ref = points_a[:, 0].squeeze(), points_a[:, 1].squeeze()
    x, y = points_b[:, 0].squeeze(), points_b[:, 1].squeeze()
    delta_x = x - x_ref
    delta_y = y_ref - y

    angle_radians = torch.atan2(delta_y, delta_x)
    angle_degrees = torch.rad2deg(angle_radians)
    return angle_degrees


def get_key_frame(data):
    # Find local maximum and minimum indices
    local_max_indices = []
    local_min_indices = []

    if data[1] > data[0]:
        local_min_indices.append(0)
    elif data[1] < data[0]:
        local_max_indices.append(0)

    for i in range(2, len(data) - 2):
        if data[i - 2] < data[i] > data[i + 2]:
            local_max_indices.append(i)
        elif data[i - 2] > data[i] < data[i + 2]:
            local_min_indices.append(i)
    if len(local_min_indices) > 0:
        local_min_indices = find_non_successive_integers(torch.tensor(local_min_indices))
    if len(local_max_indices) > 0:
        local_max_indices = find_non_successive_integers(torch.tensor(local_max_indices))

    return torch.tensor(local_min_indices), torch.tensor(local_max_indices)


def get_gradient_change_key_frame_batch(data_batch):
    # Find local maximum and minimum indices
    local_min_frames = []
    local_max_frames = []
    for data in data_batch:
        local_min, local_max = get_key_frame(data)
        local_min_frames.append(local_min.tolist())
        local_max_frames.append(local_max.tolist())
    key_frames = {'local_min': local_min_frames, 'local_max': local_max_frames}
    return key_frames


def find_non_successive_integers(numbers):
    non_successive_values = []
    if len(numbers) == 1:
        return numbers
    for i in range(len(numbers) - 1):
        if numbers[i] + 1 != numbers[i + 1]:
            non_successive_values.append(numbers[i])

    # Check the last element
    try:
        if len(numbers) > 0 and numbers[-1] != numbers[-2] + 1:
            non_successive_values.append(numbers[-1])
    except IndexError:
        print('')

    return torch.tensor(non_successive_values).to(numbers.device)


def find_non_repeat_integers(numbers):
    non_repeat_values = []
    non_repeat_indices = []
    if len(numbers) == 1:
        return numbers, torch.tensor([0])
    for i in range(len(numbers) - 1):
        if numbers[i] != numbers[i + 1]:
            non_repeat_values.append(numbers[i])
            non_repeat_indices.append(i)
    # Check the last element
    if len(numbers) > 0 and numbers[-1] != numbers[-2]:
        non_repeat_values.append(numbers[-1])
        non_repeat_indices.append(len(numbers) - 1)
    return torch.tensor(non_repeat_values).to(numbers.device), torch.tensor(non_repeat_indices).to(numbers.device)


def get_intersect_key_frames(o_i, data, touchable, movable, score):
    mask_a = data[:, 0, :-4].sum(dim=1) > 0
    mask_b = data[:, 1, :-4].sum(dim=1) > 0
    mask = mask_a * mask_b
    x_dist_close_data = torch.nan
    y_dist_close_data = torch.nan

    width_a = data[mask, 0, -4].mean()
    height_a = data[mask, 0, -3].mean()
    width_b = data[mask, 1, -4].mean()
    height_b = data[mask, 1, -3].mean()

    if mask.sum() > 0:
        x_dist_ab_min = min(width_a, width_b)
        y_dist_ab_min = min(height_a, height_b)
        x_dist_ab = torch.abs(data[:, 0, -2] - data[:, 1, -2])
        y_dist_ab = torch.abs(data[:, 0, -1] - data[:, 1, -1])
        mask_x_close = (x_dist_ab < x_dist_ab_min) * mask
        mask_y_close = (y_dist_ab < y_dist_ab_min) * mask

        dist_close_moments_x = torch.arange((len(data))).to(mask.device)[mask_x_close]
        dist_close_moments_y = torch.arange((len(data))).to(mask.device)[mask_y_close]
        if len(dist_close_moments_x) > 0:
            other_moments = find_non_successive_integers(dist_close_moments_x)
            if len(other_moments) > 0:
                x_dist_close_state_indices = torch.cat((dist_close_moments_x[0:1], other_moments), dim=0)
                dist_value_x_other = torch.cat((x_dist_ab[dist_close_moments_x[0:1]], x_dist_ab[other_moments]), dim=0)
                dist_value_y_other = torch.cat((y_dist_ab[dist_close_moments_x[0:1]], y_dist_ab[other_moments]), dim=0)
            else:
                x_dist_close_state_indices = dist_close_moments_x[0].reshape(-1)
                dist_value_x_other = x_dist_ab[dist_close_moments_x[0]].reshape(-1)
                dist_value_y_other = y_dist_ab[dist_close_moments_x[0]].reshape(-1)

            fd_objs = torch.tensor([o_i] * len(x_dist_close_state_indices)).unsqueeze(1).to(data.device)
            fd_states = x_dist_close_state_indices.unsqueeze(1)
            fd_x_dists = dist_value_x_other.unsqueeze(1)

            fd_y_dists = dist_value_y_other.unsqueeze(1)
            x_dist_close_data = torch.cat((fd_objs, fd_states, fd_x_dists, fd_y_dists),
                                          dim=1).reshape(-1, 4)
        if len(dist_close_moments_y) > 0:
            other_moments = find_non_successive_integers(dist_close_moments_y)
            if len(other_moments) > 0:
                y_dist_close_state_indices = torch.cat((dist_close_moments_y[0:1], other_moments), dim=0)
                dist_value_x_other = torch.cat((x_dist_ab[dist_close_moments_y[0:1]], x_dist_ab[other_moments]), dim=0)
                dist_value_y_other = torch.cat((y_dist_ab[dist_close_moments_y[0:1]], y_dist_ab[other_moments]), dim=0)
            else:
                y_dist_close_state_indices = dist_close_moments_y[0].reshape(-1)
                dist_value_x_other = x_dist_ab[dist_close_moments_y[0]].reshape(-1)
                dist_value_y_other = y_dist_ab[dist_close_moments_y[0]].reshape(-1)
            fd_objs = torch.tensor([o_i] * len(y_dist_close_state_indices)).unsqueeze(1).to(data.device)
            fd_states = y_dist_close_state_indices.unsqueeze(1)
            fd_x_dists = dist_value_x_other.unsqueeze(1)
            fd_y_dists = dist_value_y_other.unsqueeze(1)
            y_dist_close_data = torch.cat((fd_objs, fd_states, fd_x_dists, fd_y_dists),
                                          dim=1).reshape(-1, 4)

    return x_dist_close_data, y_dist_close_data, width_b, height_b


def get_intersect_sequence(dist_min_moments):
    dist_frames, indices = dist_min_moments[:, 1].sort()

    dist_min_moments_non_repeat, dnnbr_indices = find_non_repeat_integers(dist_min_moments[indices.reshape(-1), 1])

    return dist_min_moments_non_repeat


def get_obj_types(states, rewards):
    obj_pos_var, _ = torch.var_mean(states[:, :, -2:].sum(dim=-1).permute(1, 0), dim=-1)
    movable_obj = obj_pos_var > 0

    return movable_obj


def get_common_rows(data_a, data_b):
    # Use broadcasting to compare all pairs of rows
    equal_rows = torch.all(data_a.unsqueeze(1) == data_b.unsqueeze(0), dim=2)  # shape: (100, 86)
    # Find indices where rows are equal
    row_indices, col_indices = torch.nonzero(equal_rows, as_tuple=True)
    # Extract common rows
    row_common = data_a[row_indices]
    return row_indices


def get_diff_rows(data_a, data_b):
    diff_row = []
    diff_row_indices = []
    for r_i, row in enumerate(data_a):
        if (row == data_b).sum(dim=1).max() < 2:
            diff_row_indices.append(r_i)
            diff_row.append(row.tolist())
    diff_row = math_utils.closest_one_percent(torch.tensor(diff_row), 0.01)
    return diff_row, diff_row_indices


def state2analysis_tensor_boxing(states, obj_a_id, obj_b_id):
    obj_ab_dir = math_utils.closest_multiple_of_45(get_ab_dir(states, obj_a_id, obj_b_id)).reshape(-1)
    obj_velocities = get_state_velo(states)
    obj_velocities[obj_velocities > 0.2] = 0
    obj_velo_dir = math_utils.closest_one_percent(math_utils.get_velo_dir(obj_velocities), 0.01)
    states = math_utils.closest_one_percent(states, 0.01)
    state_tensors = torch.zeros(len(states), 7).to(states.device)
    for s_i in range(states.shape[0]):
        # pos x dist
        state_tensors[s_i, [0]] = torch.abs(states[s_i, obj_a_id, -2:-1] - states[s_i, obj_b_id, -2:-1])
        # pos y dist
        state_tensors[s_i, [1]] = torch.abs(states[s_i, obj_a_id, -1:] - states[s_i, obj_b_id, -1:])
        # left arm length
        state_tensors[s_i, [2]] = torch.abs(states[s_i, obj_a_id, -4])
        # right arm length
        state_tensors[s_i, [3]] = torch.abs(states[s_i, obj_a_id, -3])
        # va_dir
        state_tensors[s_i, [4]] = obj_velo_dir[s_i, obj_a_id]
        # vb_dir
        state_tensors[s_i, [5]] = obj_velo_dir[s_i, obj_b_id]
        # dir_ab
        state_tensors[s_i, [6]] = obj_ab_dir[s_i]

    state_tensors = math_utils.closest_one_percent(state_tensors, 0.01)
    return state_tensors


def state2analysis_tensor_pong(states, obj_a_id, obj_b_id):
    obj_ab_dir = math_utils.closest_multiple_of_45(get_ab_dir(states, obj_a_id, obj_b_id)).reshape(-1)
    obj_velocities = get_state_velo(states)
    obj_velocities[obj_velocities > 0.2] = 0
    obj_velo_dir = math_utils.closest_one_percent(math_utils.get_velo_dir(obj_velocities), 0.01)
    states = math_utils.closest_one_percent(states, 0.01)
    state_tensors = torch.zeros(len(states), 5).to(states.device)
    for s_i in range(states.shape[0]):
        state_tensors[s_i, [0]] = torch.abs(states[s_i, obj_a_id, -2:-1] - states[s_i, obj_b_id, -2:-1])
        state_tensors[s_i, [1]] = torch.abs(states[s_i, obj_a_id, -1:] - states[s_i, obj_b_id, -1:])
        state_tensors[s_i, [2]] = obj_velo_dir[s_i, obj_a_id]
        state_tensors[s_i, [3]] = obj_velo_dir[s_i, obj_b_id]
        state_tensors[s_i, [4]] = obj_ab_dir[s_i]
    state_tensors = math_utils.closest_one_percent(state_tensors, 0.01)
    return state_tensors


def state2analysis_tensor_fishing_derby(states):
    obj_ab_dir = math_utils.closest_multiple_of_45(get_ab_dir(states)).reshape(-1)
    obj_velocities = get_state_velo(states)
    obj_velocities[obj_velocities > 0.2] = 0
    obj_velo_dir = math_utils.closest_one_percent(math_utils.get_velo_dir(obj_velocities), 0.01)
    states = math_utils.closest_one_percent(states, 0.01)
    state_tensors = torch.zeros(len(states), 5).to(states.device)
    for s_i in range(states.shape[0]):
        state_tensors[s_i, [0]] = torch.abs(states[s_i, obj_a_id, -2:-1] - states[s_i, obj_b_id, -2:-1])
        state_tensors[s_i, [1]] = torch.abs(states[s_i, obj_a_id, -1:] - states[s_i, obj_b_id, -1:])
        state_tensors[s_i, [2]] = obj_velo_dir[s_i, obj_a_id]
        state_tensors[s_i, [3]] = obj_velo_dir[s_i, obj_b_id]
        state_tensors[s_i, [4]] = obj_ab_dir[s_i]
    state_tensors = math_utils.closest_one_percent(state_tensors, 0.01)
    return state_tensors


def state2analysis_tensor_kangaroo(states):
    a_i = 0  # player id is 0
    obj_num = states.shape[1]
    states = math_utils.closest_one_percent(states, 0.01)
    state_tensors = torch.zeros((obj_num - 1) * 2, len(states)).to(states.device)
    combinations = list(product(list(range(1, obj_num)), [-2, -1]))

    for c_i in range(len(combinations)):
        b_i, p_i = combinations[c_i]
        mask = states[:, b_i, :-2].sum(dim=1) > 0
        state_tensors[c_i, mask] = torch.abs(states[mask, a_i, p_i] - states[mask, b_i, p_i])
        state_tensors = math_utils.closest_one_percent(state_tensors, 0.01)

    state_tensors = torch.cat((states[:, 0:1, -2].permute(1, 0), states[:, 0:1, -1].permute(1, 0), state_tensors),
                              dim=0)
    return state_tensors


def state2analysis_tensor(states, obj_a_id, obj_b_id):
    obj_ab_dir = math_utils.closest_multiple_of_45(get_ab_dir(states, obj_a_id, obj_b_id)).reshape(-1)
    obj_velocities = get_state_velo(states)
    obj_velocities[obj_velocities > 0.2] = 0
    obj_velo_dir = math_utils.closest_one_percent(math_utils.get_velo_dir(obj_velocities), 0.01)
    states = math_utils.closest_one_percent(states, 0.01)
    state_tensors = torch.zeros(len(states), 5).to(states.device)
    for s_i in range(states.shape[0]):
        state_tensors[s_i, [0]] = torch.abs(states[s_i, obj_a_id, -2:-1] - states[s_i, obj_b_id, -2:-1])
        state_tensors[s_i, [1]] = torch.abs(states[s_i, obj_a_id, -1:] - states[s_i, obj_b_id, -1:])
        state_tensors[s_i, [2]] = obj_velo_dir[s_i, obj_a_id]
        state_tensors[s_i, [3]] = obj_velo_dir[s_i, obj_b_id]
        state_tensors[s_i, [4]] = obj_ab_dir[s_i]
    state_tensors = math_utils.closest_one_percent(state_tensors, 0.01)
    return state_tensors


def state2pos_tensor(states, obj_a_id, obj_b_id):
    obj_velocities = get_state_velo(states)
    obj_velocities[obj_velocities > 0.2] = 0
    obj_velo_dir = math_utils.closest_one_percent(math_utils.get_velo_dir(obj_velocities), 0.01)
    states = math_utils.closest_one_percent(states, 0.01)
    state_tensors = torch.zeros(len(states), 3).to(states.device)
    for s_i in range(states.shape[0]):
        state_tensors[s_i, [0]] = torch.abs(states[s_i, obj_a_id, -2:-1])
        state_tensors[s_i, [1]] = torch.abs(states[s_i, obj_a_id, -1:])
        state_tensors[s_i, [2]] = obj_velo_dir[s_i, obj_a_id]
    state_tensors = math_utils.closest_one_percent(state_tensors, 0.01)
    return state_tensors


def stat_frame_bahavior(frame_state_tensors, rewards, actions):
    pos_tensors = frame_state_tensors[torch.nonzero(rewards > 0, as_tuple=True)[0]]
    pos_actions = actions[torch.nonzero(rewards > 0, as_tuple=True)[0]]
    neg_tensors = frame_state_tensors[torch.nonzero(rewards < 0, as_tuple=True)[0]]
    neg_actions = actions[torch.nonzero(rewards < 0, as_tuple=True)[0]]
    combs = [list(combination) for i in range(2, len(range(pos_tensors.shape[1])) + 1) for combination in
             combinations(list(range(neg_tensors.shape[1])), i)]
    learned_data = []
    for index_comb in combs:
        pos_data = pos_tensors[:, index_comb].unique(dim=0)
        neg_data = neg_tensors[:, index_comb].unique(dim=0)
        pos_only_data, pos_only_indices = get_diff_rows(pos_data, neg_data)
        if len(pos_only_data) > 0:
            pos_only_data_and_action = torch.cat((pos_only_data, pos_actions[pos_only_indices].unsqueeze(1)), dim=1)
            learned_data.append([index_comb, pos_only_data_and_action.tolist()])
            mask = torch.ones(pos_tensors.size(0), dtype=torch.bool)
            mask[pos_only_indices] = 0
            pos_tensors = pos_tensors[mask]
            pos_actions = pos_actions[mask]
    return learned_data


def stat_reward_behaviors(state_tensors, key_frames, actions, rewards):
    discount_rewards = discounted_rewards(rewards, gamma=0.99, alignment=0.01)
    frames_min, frames_max = key_frames
    learned_data = []
    for p_i in range(len(frames_min)):
        frames = frames_min[p_i]
        frame_learned_data = stat_frame_bahavior(state_tensors[frames], discount_rewards[frames], actions[frames])
        learned_data.append(frame_learned_data)
    return learned_data


def stat_pf_behaviors(state_tensors, key_frames):
    beh_least_sample_num = 30
    prop_num = state_tensors.shape[1]
    prop_combs = math_utils.all_subsets(list(range(state_tensors.shape[1])))
    pf_behs = []
    for key_frame_type, key_frame_indices in key_frames.items():
        pf_data = []
        all_comb_data = []
        all_comb_data_indcies = []
        all_comb_data_state_indices = []
        for prop_indices in prop_combs:
            frame_indices = [key_frame_indices[prop] for prop in prop_indices]
            common_frame_indices = math_utils.common_elements(frame_indices)
            key_frame_tensors = state_tensors[common_frame_indices]
            key_data = key_frame_tensors[:, prop_indices]
            unique_key_data, unique_key_counts = key_data.unique(dim=0, return_counts=True)
            unique_key_data_best = unique_key_data[unique_key_counts > beh_least_sample_num].tolist()
            if len(unique_key_data_best) > 0:
                pf_data.append([prop_indices, unique_key_data_best])
                for each_unique_key in unique_key_data_best:
                    all_comb_data.append(each_unique_key)
                    all_comb_data_indcies.append(prop_indices)
                    all_comb_data_state_indices.append(common_frame_indices)
        # remove the repeat data
        learned_data = []
        for p_i in range(len(all_comb_data) - 1):
            is_repeat = False
            for p_j in range(p_i + 1, len(all_comb_data)):
                if math_utils.is_sublist(all_comb_data[p_i], all_comb_data[p_j]):
                    if math_utils.is_sublist(all_comb_data_indcies[p_i], all_comb_data_indcies[p_j]):
                        is_repeat = True
            if not is_repeat:
                learned_data.append(
                    [all_comb_data_indcies[p_i], all_comb_data[p_i], all_comb_data_state_indices[p_i], key_frame_type])
        learned_data.append(
            [all_comb_data_indcies[-1], all_comb_data[-1], all_comb_data_state_indices[-1], key_frame_type])
        pf_behs += learned_data
    return pf_behs


def stat_o2o_action(args, states, actions):
    # remove non-removed states
    action_types = actions.unique().tolist()
    delta_actions = actions[:-1]
    actions_delta = {}
    state_delta_tensors = math_utils.closest_one_percent(states[1:] - states[:-1], 0.01)
    state_tensors = state_delta_tensors[state_delta_tensors[:, 2] != 0]
    delta_actions = delta_actions[state_delta_tensors[:, 2] != 0]

    for action_type in action_types:
        action_state_tensors = state_tensors[delta_actions == action_type]

        x = math_utils.remove_outliers_iqr(action_state_tensors[:, 0])
        y = math_utils.remove_outliers_iqr(action_state_tensors[:, 1])
        dir = math_utils.remove_outliers_iqr(action_state_tensors[:, 2])

        x_mean = x.median()
        y_mean = y.median()
        dir_mean = dir.median()
        action_delta = [x_mean.tolist(), y_mean.tolist(), dir_mean.tolist()]
        actions_delta[action_type] = action_delta
        draw_utils.plot_compare_line_chart(action_state_tensors.permute(1, 0).tolist(),
                                           args.check_point_path / "o2o", f'act_delta_{action_type}',
                                           (30, 10), row_names=['dx', 'dy', 'dir_a'])
        draw_utils.plot_compare_line_chart([x.tolist(), y.tolist(), dir.tolist()],
                                           args.check_point_path / "o2o", f'act_delta_{action_type}_iqr',
                                           (30, 10), row_names=['dx', 'dy', 'dir_a'])

    return actions_delta


def text_from_tensor(o2o_data, state_tensors, prop_explain):
    explain_text = ""
    o2o_behs = []
    dist_o2o_behs = []
    dist_to_o2o_behs = []
    next_possile_beh_explain_text = []
    for beh_i, o2o_beh in enumerate(o2o_data):
        beh_type = o2o_beh[3]
        game_values = state_tensors[-1, o2o_beh[0]]
        beh_values = torch.tensor(o2o_beh[1]).to(state_tensors.device)

        value = ["{:.2f}".format(num) for num in o2o_beh[1]]
        prop_explains = [prop_explain[prop] for prop in o2o_beh[0]]
        if (game_values == beh_values).prod().bool():
            explain_text += f"{prop_explains}_{value}_{len(o2o_beh[2])}\n"
            next_possile_beh_explain_text.append("")
            o2o_behs.append(beh_i)
            dist_to_o2o_behs.append(0)
            dist_o2o_behs.append(torch.tensor([0]))


        else:
            dist = game_values - beh_values
            dist_value = ["{:.2f}".format(num) for num in dist]
            next_possile_beh_explain_text.append(
                f"goto_{beh_type}_dist_{dist_value}_{prop_explains}_{value}_{len(o2o_beh[2])}\n")
            dist_to_o2o_behs.append(torch.abs(dist).sum().tolist())
            dist_o2o_behs.append(torch.abs(dist).sum())

    dist_to_o2o_behs = torch.tensor(dist_o2o_behs).abs()

    return dist_to_o2o_behs, next_possile_beh_explain_text


def game_explain(state, last_state, last2nd_state, o2o_data):
    explain_text = ""
    prop_explain = {0: 'dx', 1: 'dy', 2: 'va', 3: 'vb', 4: 'dir_ab'}
    state3 = torch.cat((torch.tensor(last2nd_state).unsqueeze(0),
                        torch.tensor(last_state).unsqueeze(0),
                        torch.tensor(state).unsqueeze(0)), dim=0)
    state_tensors = state2analysis_tensor(state3, 0, 1)
    for o2o_beh in o2o_data:
        game_values = state_tensors[-1, o2o_beh[0]]
        beh_values = torch.tensor(o2o_beh[1])
        if (game_values == beh_values).prod().bool():
            beh_type = o2o_beh[3]
            game_last_values = state_tensors[-2, o2o_beh[0]]
            if beh_type == 'local_min' and (game_last_values > game_values).prod().bool():
                value = ["{:.2f}".format(num) for num in o2o_beh[1]]
                prop_explains = [prop_explain[prop] for prop in o2o_beh[0]]
                explain_text += f"min_{prop_explains}_{value}_{len(o2o_beh[2])}\n"
            elif beh_type == 'local_max' and (game_last_values < game_values).prod().bool():
                value = ["{:.2f}".format(num) for num in o2o_beh[1]]
                prop_explains = [prop_explain[prop] for prop in o2o_beh[0]]
                explain_text += f"max_{prop_explains}_{value}_{len(o2o_beh[2])}\n"
    return explain_text


def visual_state_tensors(args, state_tensors):
    row_names = ["reward"]
    for r_i in range(0, len(args.row_names)):
        row_names.append(f"{args.row_names[r_i]}_x")
        row_names.append(f"{args.row_names[r_i]}_y")

    draw_utils.plot_compare_line_chart(state_tensors.tolist(),
                                       path=args.check_point_path / "o2o",
                                       name=f"st_{args.m}", figsize=(30, 100),
                                       pos_color="orange",
                                       neg_color="blue",
                                       row_names=row_names)


def find_direction_ranges(positions):
    increase_indices = []
    decrease_indices = []
    for i in range(len(positions) - 1):
        if positions[i + 1] > positions[i]:
            increase_indices.append(i)
        elif positions[i + 1] < positions[i]:
            decrease_indices.append(i)
    return increase_indices, decrease_indices


def find_closest_obj_over_states(states, axis):
    dist_means = []
    for obj_i in range(1, states.shape[1]):
        mask_0 = states[:, 0, :-4].sum(axis=-1) > 0
        mask_1 = states[:, obj_i, :-4].sum(axis=-1) > 0
        mask = mask_0 * mask_1
        dist = torch.abs(states[mask, 0, axis] - states[mask, obj_i, axis])
        var, mean = torch.var_mean(dist)
        dist_means.append(mean)
    _, closest_index = torch.tensor(dist_means).sort()
    return closest_index + 1


def reason_shiftness(args, states):
    x_posisions = states[:, 0, -2]
    x_positions_smooth = math_utils.smooth_filter(x_posisions, window_size=50)
    x_increase_indices, x_decrease_indices = find_direction_ranges(x_posisions)
    states_x_increase = states[x_increase_indices]
    states_x_decrease = states[x_decrease_indices]
    dx_pos_indices = find_closest_obj_over_states(states_x_increase, -1)
    dx_neg_indices = find_closest_obj_over_states(states_x_decrease, -1)

    y_posisions = states[:, 0, -1]
    y_positions_smooth = math_utils.smooth_filter(y_posisions, window_size=300)
    draw_utils.plot_line_chart(y_positions_smooth.unsqueeze(0).to("cpu"), path=".",
                               labels=["pos_y"], title="position_y")
    y_increase_indices, y_decrease_indices = find_direction_ranges(y_positions_smooth)
    states_y_increase = states[y_increase_indices]
    states_y_decrease = states[y_decrease_indices]
    dy_pos_indices = find_closest_obj_over_states(states_y_increase, -2)
    dy_neg_indices = find_closest_obj_over_states(states_y_decrease, -2)

    for i in range(len(dx_pos_indices)):
        print(f"{i}: \n "
              f"dx pos {args.row_names[dx_pos_indices[i]]}, \n"
              f"dx neg {args.row_names[dx_neg_indices[i]]}, \n"
              f"dy pos {args.row_names[dy_pos_indices[i]]}, \n"
              f"dy neg {args.row_names[dy_pos_indices[i]]}. ")
    rulers = {"decrease": {-2: dx_neg_indices.tolist(), -1: dy_neg_indices.tolist()},
              "increase": {-2: dx_pos_indices.tolist(), -1: dy_neg_indices.tolist()}}

    return rulers


def batch_calculate_overlap(boxes1, boxes2):
    """
    Calculate the overlap between multiple pairs of rectangle boxes in batch.

    Parameters:
        boxes1 (torch.Tensor): Coordinates of the first set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].
        boxes2 (torch.Tensor): Coordinates of the second set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].

    Returns:
        overlap_areas (torch.Tensor): The overlap areas between each pair of boxes in the batch.
                                       If the boxes do not overlap, return 0.
        overlap_ratios (torch.Tensor): The ratios of overlap areas to the areas of the smaller boxes in each pair.
    """
    # Calculate the coordinates of the intersection rectangle for each pair of boxes
    x1_inter = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y1_inter = torch.minimum(boxes1[:, 1], boxes2[:, 1])
    x2_inter = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y2_inter = torch.maximum(boxes1[:, 3], boxes2[:, 3])

    # Calculate the width and height of the intersection rectangle for each pair of boxes
    width_inter = torch.maximum(torch.zeros_like(x1_inter), x2_inter - x1_inter)
    height_inter = torch.maximum(torch.zeros_like(y1_inter), y1_inter - y2_inter)

    # Calculate the area of intersection rectangle for each pair of boxes
    area_inter = width_inter * height_inter

    # Calculate the overlap areas and overlap ratios for each pair of boxes
    overlap = area_inter > 0
    return overlap


def batch_calculate_left(boxes1, boxes2):
    """
    Determine if boxes1 are on the left side of boxes2 and their positions align on the y-axis.

    Parameters:
        boxes1 (torch.Tensor): Coordinates of the first set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].
        boxes2 (torch.Tensor): Coordinates of the second set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].

    Returns:
        is_on_left (torch.Tensor): A boolean tensor indicating if boxes1 are on the left side of boxes2
                                    and their positions align on the y-axis for each pair.
    """
    # Extract y-coordinates of the top and bottom edges of boxes1 and boxes2
    bottom_edge_boxes1 = boxes1[:, 1]
    bottom_edge_boxes2 = boxes2[:, 1]
    top_edge_boxes1 = boxes1[:, 3]
    top_edge_boxes2 = boxes2[:, 3]

    align_y_axis = ((bottom_edge_boxes1 < bottom_edge_boxes2) & (bottom_edge_boxes1 > top_edge_boxes2)) | \
                   ((bottom_edge_boxes2 < bottom_edge_boxes1) & (bottom_edge_boxes2 > top_edge_boxes1))

    # Extract x-coordinate of the right edge of boxes1 and the left edge of boxes2
    right_edge_boxes1 = boxes1[:, 2]
    left_edge_boxes2 = boxes2[:, 0]

    # Determine if boxes1 are on the left side of boxes2 and their positions align on the y-axis
    is_on_left = align_y_axis & (right_edge_boxes1 < left_edge_boxes2)

    return is_on_left


def batch_calculate_right(boxes1, boxes2):
    """
    Determine if boxes1 are on the left side of boxes2 and their positions align on the y-axis.

    Parameters:
        boxes1 (torch.Tensor): Coordinates of the first set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].
        boxes2 (torch.Tensor): Coordinates of the second set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].

    Returns:
        is_on_left (torch.Tensor): A boolean tensor indicating if boxes1 are on the left side of boxes2
                                    and their positions align on the y-axis for each pair.
    """
    # Extract y-coordinates of the top and bottom edges of boxes1 and boxes2
    top_edge_boxes1 = boxes1[:, 3]
    top_edge_boxes2 = boxes2[:, 3]
    bottom_edge_boxes1 = boxes1[:, 1]
    bottom_edge_boxes2 = boxes2[:, 1]

    # Determine if the positions align on the y-axis
    align_y_axis = ((bottom_edge_boxes1 < bottom_edge_boxes2) & (bottom_edge_boxes1 > top_edge_boxes2)) | \
                   ((bottom_edge_boxes2 < bottom_edge_boxes1) & (bottom_edge_boxes2 > top_edge_boxes1))

    # Extract x-coordinate of the right edge of boxes1 and the left edge of boxes2
    left_edge_boxes1 = boxes1[:, 0]
    right_edge_boxes2 = boxes2[:, 2]

    # Determine if boxes1 are on the left side of boxes2 and their positions align on the y-axis
    is_on_left = align_y_axis & (left_edge_boxes1 > right_edge_boxes2)

    return is_on_left


def batch_calculate_above(boxes1, boxes2):
    """
    Determine if boxes1 are on the left side of boxes2 and their positions align on the y-axis.

    Parameters:
        boxes1 (torch.Tensor): Coordinates of the first set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].
        boxes2 (torch.Tensor): Coordinates of the second set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].

    Returns:
        is_on_left (torch.Tensor): A boolean tensor indicating if boxes1 are on the left side of boxes2
                                    and their positions align on the y-axis for each pair.
    """
    # Extract y-coordinates of the top and bottom edges of boxes1 and boxes2
    right_edge_boxes1 = boxes1[:, 2]
    right_edge_boxes2 = boxes2[:, 2]
    left_edge_boxes1 = boxes1[:, 0]
    left_edge_boxes2 = boxes2[:, 0]

    # Determine if the positions align on the y-axis
    align_x_axis = ((left_edge_boxes1 > left_edge_boxes2) & (left_edge_boxes1 < right_edge_boxes2)) | \
                   ((left_edge_boxes2 > left_edge_boxes1) & (left_edge_boxes2 < right_edge_boxes1))

    # Extract x-coordinate of the right edge of boxes1 and the left edge of boxes2
    bottom_edge_boxes1 = boxes1[:, 1]
    top_edge_boxes2 = boxes2[:, 3]

    # Determine if boxes1 are on the left side of boxes2 and their positions align on the y-axis
    is_above = align_x_axis & (bottom_edge_boxes1 < top_edge_boxes2)

    return is_above


def batch_calculate_below(boxes1, boxes2):
    """
    Determine if boxes1 are on the left side of boxes2 and their positions align on the y-axis.

    Parameters:
        boxes1 (torch.Tensor): Coordinates of the first set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].
        boxes2 (torch.Tensor): Coordinates of the second set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].

    Returns:
        is_on_left (torch.Tensor): A boolean tensor indicating if boxes1 are on the left side of boxes2
                                    and their positions align on the y-axis for each pair.
    """
    # Extract y-coordinates of the top and bottom edges of boxes1 and boxes2
    right_edge_boxes1 = boxes1[:, 2]
    right_edge_boxes2 = boxes2[:, 2]
    left_edge_boxes1 = boxes1[:, 0]
    left_edge_boxes2 = boxes2[:, 0]

    # Determine if the positions align on the y-axis
    align_x_axis = ((left_edge_boxes1 > left_edge_boxes2) & (left_edge_boxes1 < right_edge_boxes2)) | \
                   ((left_edge_boxes2 > left_edge_boxes1) & (left_edge_boxes2 < right_edge_boxes1))

    # Extract x-coordinate of the right edge of boxes1 and the left edge of boxes2
    top_edge_boxes1 = boxes1[:, 1]
    bottom_edge_boxes2 = boxes2[:, 3]

    # Determine if boxes1 are on the left side of boxes2 and their positions align on the y-axis
    is_below = align_x_axis & (top_edge_boxes1 > bottom_edge_boxes2)

    return is_below


def are_distances_aligned(boxes1, boxes2):
    """
    Check if the distances between the centers of two boxes align with x and y axes and are less than given thresholds.

    Parameters:
        boxes1 (torch.Tensor): Coordinates of the first set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].
        boxes2 (torch.Tensor): Coordinates of the second set of rectangle boxes with shape (batch_size, 4).
                               Each row represents a box in the format [x1, y1, x2, y2].
        th_x (float): Threshold for the x-axis distance between box centers.
        th_y (float): Threshold for the y-axis distance between box centers.

    Returns:
        are_aligned (torch.Tensor): A boolean tensor indicating if the distances align with x and y axes
                                     and are less than the given thresholds for each pair of boxes.
    """
    x1_mean = (boxes1[:, 2] - boxes1[:, 0]).mean()
    x2_mean = (boxes2[:, 2] - boxes2[:, 0]).mean()
    y1_mean = (boxes1[:, 1] - boxes1[:, 3]).mean()
    y2_mean = (boxes2[:, 1] - boxes2[:, 3]).mean()

    th_x, th_y = (x1_mean + x2_mean), (y1_mean + y2_mean)

    # Calculate the center coordinates of each box
    center_x1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
    center_y1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
    center_x2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
    center_y2 = (boxes2[:, 1] + boxes2[:, 3]) / 2

    # Calculate the distances between the centers of the boxes along x and y axes
    dist_x = torch.abs(center_x1 - center_x2)
    dist_y = torch.abs(center_y1 - center_y2)

    # Check if the distances align with x and y axes and are less than the given thresholds
    are_aligned = (dist_x <= th_x) & (dist_y <= th_y)

    return are_aligned


def extract_positive_behaviors(args, states, rewards, actions):
    positive_states = states[rewards > 0]
    positive_actions = actions[rewards > 0]
    positive_behaviors = []
    p_texts = ["overlap", "left_of", "right_of", "above_of", "below_of", "close_to"]
    used_relation_b = []
    # print out
    for state_i in range(len(positive_states)):
        for o_i in range(len(positive_states[state_i])):
            obj_a = f"{args.row_names[0]}_"
            relation = ""
            obj_b = f"{args.row_names[o_i]}"
            for p_i in range(len(positive_states[state_i][o_i])):
                if positive_states[state_i, o_i, p_i]:
                    relation += p_texts[p_i] + "_"
            if relation != "" and relation + obj_b not in used_relation_b:
                used_relation_b.append(relation + obj_b)
                positive_behaviors.append([o_i, relation[:-1], positive_actions[state_i], "kill"])
                positive_behaviors.append([o_i, relation[:-1], positive_actions[state_i], "align"])
                positive_behaviors.append([o_i, relation[:-1], positive_actions[state_i], "avoid"])
                print(f"positive {state_i}: {obj_a + relation + obj_b}, act: {positive_actions[state_i]}")

    return positive_behaviors


def extract_negative_behaviors(args, states, rewards, actions):
    negative_states = states[rewards < 0]
    negative_actions = actions[rewards < 0]
    negative_behaviors = []
    p_texts = ["overlap", "left_of", "right_of", "above_of", "below_of", "close_to"]
    used_relation_b = []

    # print out
    for state_i in range(len(negative_states)):
        for o_i in range(len(negative_states[state_i])):
            obj_a = f"{args.row_names[0]}_"
            relation = ""
            obj_b = f"{args.row_names[o_i]}"
            for p_i in range(len(negative_states[state_i][o_i])):
                if negative_states[state_i, o_i, p_i]:
                    relation += p_texts[p_i] + "_"
            if relation != "" and relation + obj_b not in used_relation_b:
                used_relation_b.append(relation + obj_b)
                negative_behaviors.append([o_i, relation[:-1], negative_actions[state_i], "kill"])
                negative_behaviors.append([o_i, relation[:-1], negative_actions[state_i], "avoid"])
                print(f"negative {state_i}: {obj_a + relation + obj_b}, act: {negative_actions[state_i]}")

    return negative_behaviors


def get_state_symbolic_data(states):
    # two aries: collide frame, collide object
    player_is_overlap_with = torch.zeros(states.shape[0], states.shape[1], dtype=torch.bool).to(states.device)
    player_is_left_of = torch.zeros(states.shape[0], states.shape[1], dtype=torch.bool).to(states.device)
    player_is_right_of = torch.zeros(states.shape[0], states.shape[1], dtype=torch.bool).to(states.device)
    player_is_above_of = torch.zeros(states.shape[0], states.shape[1], dtype=torch.bool).to(states.device)
    player_is_below_of = torch.zeros(states.shape[0], states.shape[1], dtype=torch.bool).to(states.device)
    player_is_close_to = torch.zeros(states.shape[0], states.shape[1], dtype=torch.bool).to(states.device)

    for o_i in range(1, states.shape[1]):
        mask_player = states[:, 0, :-6].sum(dim=-1) > 0
        mask_oi = states[:, o_i, :-6].sum(dim=-1) > 0
        mask = mask_player & mask_oi
        player_position = states[mask, 0, -6:-2]
        others_position = states[mask, o_i, -6:-2]
        player_is_overlap_with[mask, o_i] = batch_calculate_overlap(player_position, others_position)
        player_is_left_of[mask, o_i] = batch_calculate_left(player_position, others_position)
        player_is_right_of[mask, o_i] = batch_calculate_right(player_position, others_position)
        player_is_above_of[mask, o_i] = batch_calculate_above(player_position, others_position)
        player_is_below_of[mask, o_i] = batch_calculate_below(player_position, others_position)
        player_is_close_to[mask, o_i] = are_distances_aligned(player_position, others_position)

    player_data = torch.cat((
        player_is_overlap_with.unsqueeze(2),
        player_is_left_of.unsqueeze(2),
        player_is_right_of.unsqueeze(2),
        player_is_above_of.unsqueeze(2),
        player_is_below_of.unsqueeze(2),
        player_is_close_to.unsqueeze(2),
    ), dim=2)
    return player_data


def reason_o2o_states(args, states, actions):
    rewards = torch.zeros(len(states))
    rewards[[0]] = 100
    rewards[[629]] = -100
    player_data = get_state_symbolic_data(states)
    positive_behaviors = extract_positive_behaviors(args, player_data, rewards, actions)
    negative_behaviors = extract_negative_behaviors(args, player_data, rewards, actions)

    return positive_behaviors, negative_behaviors


def min_value_greater_than(tensor, threshold):
    # Mask the tensor where values are greater than -0.05
    masked_tensor = tensor[tensor > threshold]

    # Get the minimum value and its index
    min_value, min_index = torch.min(masked_tensor, dim=0)

    # Adjust index to account for masking
    original_index = (tensor > threshold).nonzero()[min_index][0].item()

    return min_value.item(), original_index


def determine_next_sub_object(args, agent, state, dist_now):
    if dist_now > 0:
        rulers = agent.model.shift_rulers["decrease"][str(agent.model.align_axis[0])]
    elif dist_now < 0:
        rulers = agent.model.shift_rulers["increase"][str(agent.model.align_axis[0])]
    else:
        raise ValueError
    if -1 in agent.model.align_axis:
        sub_align_axis = [-2]
    elif -2 in agent.model.align_axis:
        sub_align_axis = [-1]
    else:
        raise ValueError

    # update target object
    sub_target_type = rulers[0]

    # if target object has duplications, find the one closest with the target position
    sub_same_others = args.same_others[sub_target_type]
    dy_sub_same_others = state[0, agent.model.align_axis] - state[sub_same_others, agent.model.align_axis]
    # Mask the tensor to get non-negative values
    # Find the minimum non-negative value and its index
    try:
        if dy_sub_same_others.max() < -0.05:
            min_value, min_index = min_value_greater_than(dy_sub_same_others, -1)
        else:
            min_value, min_index = min_value_greater_than(dy_sub_same_others, -0.05)

    except IndexError:
        print("")
    # Find the original index in the complete tensor
    try:
        original_index = (dy_sub_same_others == min_value).nonzero()[0].item()
    except RuntimeError:
        print("")

    sub_target = sub_same_others[original_index]

    print(f"- Target Object: {args.row_names[agent.model.target_obj]}. \n"
          f"- Failed to align axis: {agent.model.align_axis}. \n"
          f"- Align Sub Object {args.row_names[sub_target]}. \n"
          f"- Align Sub Axis {sub_align_axis}.\n")

    return sub_target, sub_align_axis


def get_obj_wh(states):
    obj_num = states.shape[1]
    obj_whs = torch.zeros((obj_num, 2))
    for o_i in range(obj_num):
        mask = states[:, o_i, :-4].sum(dim=-1) > 0
        if mask.sum() == 0:
            obj_whs[o_i] = obj_whs[o_i - 1]
        else:
            obj_whs[o_i, 0] = states[mask, o_i, -4].mean()
            obj_whs[o_i, 1] = states[mask, o_i, -3].mean()

    return obj_whs


def reason_danger_distance(args, states, rewards):
    args.obj_whs = get_obj_wh(states)
    obj_names = args.row_names
    danger_distance = torch.zeros((len(obj_names), 2)) - 1
    player_wh = args.obj_whs[0]
    for o_i, (touchable, movable, scorable) in enumerate(args.obj_data):
        if not touchable:
            danger_distance[o_i, 0] = (player_wh[0] + args.obj_whs[o_i, 0]) * 0.5
            danger_distance[o_i, 1] = (player_wh[1] + args.obj_whs[o_i, 1]) * 0.5

    return danger_distance


def learn_surrounding_dangerous(env_args, agent, args, test_neg_beh):
    # only one object and one axis can be determined
    # determine the first coming enemy, which might not be the closest one, it depends on its position and speed
    frame_eval_num = 15
    if test_neg_beh is None:
        neg_behs = agent.negative_behaviors
    else:
        neg_behs = agent.negative_behaviors + [test_neg_beh]

    time_lefts = []
    min_objs = []
    danger_axis_list = []
    strategy_list = []
    for neg_beh in neg_behs:
        min_dist_obj, danger_axis = None, None
        enemy_id, danger_relation, danger_action, strategy = neg_beh

        past_states = torch.tensor(env_args.past_states)

        # get the positions
        enemy_ids = args.same_others[enemy_id]
        for id in enemy_ids:
            enemy_last_positions = past_states[-frame_eval_num:, id, -6:-2]
            player_position = past_states[-1:, 0, -6:-2]
            player_position = torch.repeat_interleave(player_position, frame_eval_num, dim=0)

            # check the closest dist between player and enemy within 5 next frames
            enemy_next_positions_l, speed_axis_l = get_next_positions(enemy_last_positions[:, :2])
            enemy_next_positions_r, speed_axis_r = get_next_positions(enemy_last_positions[:, 2:])
            enemy_next_positions = torch.cat((enemy_next_positions_l, enemy_next_positions_r), dim=1)

            satisfaction = get_satisfy_moment(player_position, enemy_next_positions, danger_relation)
            if satisfaction.float().sum() > 0:
                time_left = satisfaction.float().argmax()
            else:
                time_left = 1000
            time_lefts.append(time_left)
            if satisfaction.sum() > 0:
                min_dist_obj = id
                danger_axis = get_danger_axis(danger_relation)
                print(f"-(might danger) "
                      f"{danger_relation} {args.row_names[min_dist_obj]} "
                      f"after {satisfaction.float().argmax()} frames.")
                aasss = 1
            min_objs.append(min_dist_obj)
            danger_axis_list.append(danger_axis)
            strategy_list.append(strategy)
    best_id = torch.tensor(time_lefts).argmin()

    min_dist_obj = min_objs[best_id]
    danger_axis = danger_axis_list[best_id]
    strategy = strategy_list[best_id]
    return min_dist_obj, danger_axis, strategy


def determine_surrounding_dangerous(env_args, agent, args):
    # only one object and one axis can be determined
    # determine the first coming enemy, which might not be the closest one, it depends on its position and speed

    past_states = torch.tensor(env_args.past_states)
    untouchable = ~args.obj_data[:, 0]
    dist_danger = agent.model.dangerous_rulers
    danger_data = []
    collision_times = torch.zeros(past_states.shape[1]) + 1e+20
    collision_axis = torch.zeros(past_states.shape[1]) + 1e+20
    min_dist_obj, danger_axis = None, None
    frame_eval_num = 15
    for o_i in range(past_states.shape[1]):
        # skip if it is touchable
        if not untouchable[o_i]:
            continue
        # skip if it doesn't exist in last 5 frames
        if past_states[:, o_i, :-4].sum(dim=-1)[-frame_eval_num:].sum() != frame_eval_num:
            continue

        # get the positions
        enemy_last_positions = past_states[-frame_eval_num:, o_i, -2:]
        player_position = past_states[-1:, 0, -2:]
        # check the closest dist between player and enemy within 5 next frames
        enemy_next_positions, speed_axis = get_next_positions(enemy_last_positions)
        collision, collision_moment = get_collide_moment(player_position, enemy_next_positions, dist_danger[o_i])

        # if trajectory collide
        if collision:
            collision_times[o_i] = collision_moment
            if speed_axis == 0:
                collision_axis[o_i] = -1
            else:
                collision_axis[o_i] = -2

        else:
            continue
    if collision_times.min() < 1e+20:
        min_dist_obj = collision_times.argmin()
        danger_axis = collision_axis[min_dist_obj].to(torch.int)
        print(
            f"- Collide with {args.row_names[min_dist_obj]} after {collision_times.min():.1f} frames at axis {danger_axis}")
    return min_dist_obj, danger_axis


def observe_unaligned(args, agent, state):
    # keep observing
    agent.model.unaligned_frame_counter += 1
    min_move_dist = 0.04
    observe_window = 15
    dist_now = state[0, agent.model.unaligned_axis] - state[agent.model.next_target, agent.model.unaligned_axis]
    agent.model.move_history.append(state[0, agent.model.align_axis])
    if len(agent.model.move_history) > observe_window:
        move_dist = torch.abs(agent.model.move_history[-observe_window] - agent.model.move_history[-1])
        # if stop moving
        if move_dist < min_move_dist:
            # if aligning to the sub-object
            if agent.model.align_to_sub_object:
                agent.model.align_to_sub_object = False

                # if align with the target
                if dist_now.sum() < 0.02:
                    print(f"- (Success) Align with {args.row_names[agent.model.next_target]} "
                          f"at Axis {agent.model.align_axis}.\n"
                          f"- Now try to find out next Align Target.")
                # if it doesn't decrease, update the symbolic-state
                else:
                    # update aligned object
                    agent.model.align_to_sub_object = True
                    print(f"- Move distance over (param) 20 frames is {move_dist}, "
                          f"less than threshold (param) {min_move_dist:.4f} \n"
                          f"- Failed to align with {args.row_names[agent.model.next_target]} at axis "
                          f"{agent.model.align_axis}")

            # if unaligned to the target object
            elif agent.model.unaligned:
                agent.model.unaligned = False
                if dist_now.sum() > 0.02:
                    # successful unaligned with object at axis
                    print(f"- (Success) Unaligned with {args.row_names[agent.model.next_target]} "
                          f"at Axis {agent.model.align_axis}.\n"
                          f"- Now try to find out next Align Target.")
                else:
                    # update aligned object
                    agent.model.align_to_sub_object = True
                    print(f"- Move distance over (param) 20 frames is {move_dist}, "
                          f"less than threshold (param) {min_move_dist:.4f} \n"
                          f"- Failed to unaligned with {args.row_names[agent.model.next_target]} at axis "
                          f"{agent.model.unaligned_axis}")


def align_to_other_obj(args, agent, state):
    agent.model.aligning = True
    agent.model.align_frame_counter = 0
    agent.model.move_history = []

    dx = torch.abs(state[0, -2] - state[agent.model.target_obj, -2])
    dy = torch.abs(state[0, -1] - state[agent.model.target_obj, -1])
    # determine next sub aligned object
    if agent.model.align_axis == -2:
        dist_now = dx
    elif agent.model.align_axis == -1:
        dist_now = dy
    else:
        raise ValueError
    next_target, align_axis = determine_next_sub_object(args, agent, state, dist_now)

    agent.model.align_axis = align_axis
    agent.model.next_target = next_target
    print(f"- New Align Target {args.row_names[agent.model.next_target]}, Axis: {agent.model.align_axis}.\n")


def unaligned_axis(args, agent, state):
    agent.model.next_target = agent.model.unaligned_target
    dx = torch.abs(state[0, -2] - state[agent.model.unaligned_target, -2])
    dy = torch.abs(state[0, -1] - state[agent.model.unaligned_target, -1])
    axis_is_unaligned = [dx > 0.02, dy > 0.02]
    if not axis_is_unaligned[0] and dx < dy:
        agent.model.unaligned_axis = [-2]
        agent.model.dist = dx
    elif not axis_is_unaligned[1] and dy < dx:
        agent.model.unaligned_axis = [-1]
        agent.model.dist = dy
    print(f"- New Unaligned Target {args.row_names[agent.model.next_target]}, Axis: {agent.model.unaligned_axis}.\n")


# Function to predict positions at future times
def predict_positions(x0, y0, x4, y4, t):
    # Calculate velocities
    vx = (x4 - x0) / 4
    vy = (y4 - y0) / 4

    # Predict positions at future times
    xt = x4 + vx * (t - 4)
    yt = y4 + vy * (t - 4)

    return xt, yt, vx, vy


def get_next_positions(past_positions):
    # Predict positions at times 5, 6, 7, 8, and 9 for both objects
    next_positions = torch.zeros(15, 2)

    # Calculate change in position
    change_in_position = past_positions[-1] - past_positions[0]

    current_position = past_positions[-1]
    # Calculate speed
    speed = change_in_position / len(past_positions)

    for i in range(len(next_positions)):
        next_positions[i] = current_position + speed * i
    speed_axis = torch.abs(speed).argmax()
    return next_positions, speed_axis


def get_collide_moment(player_position, enemy_positions, save_dist):
    save_dist_ = save_dist.max()
    dist_e_to_p = torch.abs(enemy_positions - player_position).sum(dim=-1)
    collide = (dist_e_to_p < save_dist_).sum().bool()
    collide_frame = (dist_e_to_p < save_dist_).float().argmax()
    return collide, collide_frame


def get_danger_axis(relation):
    danger_axis = []
    if "left_of" in relation:
        danger_axis.append(-1)
    if "right_of" in relation:
        danger_axis.append(-1)
    if "below_of" in relation:
        danger_axis.append(-2)
    if "above_of" in relation:
        danger_axis.append(-2)
    if "close_to" in relation:
        danger_axis.append(-1)
        danger_axis.append(-2)
    if "overlap" in relation:
        danger_axis.append(-1)
        danger_axis.append(-2)
    danger_axis = torch.tensor(danger_axis).unique()
    return danger_axis


def get_satisfy_moment(player_position, enemy_positions, danger_relation):
    satisfaction = torch.ones(len(player_position), dtype=torch.bool)
    if "overlap" in danger_relation:
        satisfaction &= batch_calculate_overlap(player_position, enemy_positions)
    if "left_of" in danger_relation:
        satisfaction &= batch_calculate_left(player_position, enemy_positions)
    if "right_of" in danger_relation:
        satisfaction &= batch_calculate_right(player_position, enemy_positions)
    if "below_of" in danger_relation:
        satisfaction &= batch_calculate_below(player_position, enemy_positions)
    if "above_of" in danger_relation:
        satisfaction &= batch_calculate_above(player_position, enemy_positions)
    if "close_to" in danger_relation:
        satisfaction &= are_distances_aligned(player_position, enemy_positions)

    return satisfaction


def decide_deal_to_enemy(args, env_args, agent, danger_obj):
    # avoid, kill, ignore

    past_states = torch.tensor(env_args.past_states)
    dist_danger = agent.model.dangerous_rulers

    # get the positions
    decide_frame_num = 15
    enemy_last_positions = past_states[-decide_frame_num:, danger_obj, -2:]
    player_position = past_states[-1:, 0, -2:]
    # check the closest dist between player and enemy within 5 next frames
    enemy_next_positions, speed_axis = get_next_positions(enemy_last_positions)
    collision, collision_moment = get_collide_moment(player_position, enemy_next_positions, dist_danger[danger_obj])

    if collision:
        if args.obj_data[danger_obj][2]:
            decision = "kill"
        else:
            decision = "avoid"
    else:
        decision = "ignore"
    print(f"- decision: {decision} {args.row_names[danger_obj]}")
    return decision


def reason_pong(args, states, actions):
    states = torch.cat(states, dim=0)
    actions = torch.cat(actions, dim=0)
    mask_player = states[:, 0, :-6].sum(dim=-1) > 0
    for obj_i in range(1, states.shape[1]):
        mask_obj_i = states[:, obj_i, :-6].sum(dim=-1) > 0
        mask = mask_player & mask_obj_i
        op_res = states[mask, 0, -2:] - states[mask, obj_i, -2:]
        x_ticks = torch.arange(-0.1, 0.8, 0.1)
        y_ticks = torch.arange(-0.7, 0.8, 0.1)
        x_ticks_2decimal = ["{:.1f}".format(num) for num in x_ticks]
        y_ticks_2decimal = ["{:.1f}".format(num) for num in y_ticks]
        heat_data = torch.zeros((len(actions.unique()), len(y_ticks), len(x_ticks)))
        for a_i, action in enumerate(actions.unique()):
            action_mask = (actions == action) & mask
            op_res = math_utils.closest_one_percent(op_res, unit=0.1)
            values, value_counts = op_res[action_mask].unique(dim=0, return_counts=True)
            for v_i in range(len(values)):
                x_i = torch.where(values[v_i, 0] == x_ticks)[0].to(torch.int)
                y_i = torch.where(values[v_i, 1] == y_ticks)[0].to(torch.int)

                heat_data[a_i, y_i, x_i] = value_counts[v_i].to(torch.float)

        action_heat_data = heat_data.argmax(dim=0)
        draw_utils.plot_heat_map(data=action_heat_data, path=args.output_folder, name=f"{args.row_names[obj_i]}",
                                 figsize=(10, 5), row_names=y_ticks_2decimal, col_names=x_ticks_2decimal)
    return None


def extract_asterix_kinematics(args, logic_state):
    logic_state = torch.tensor(logic_state).to(args.device)
    velo = get_state_velo(logic_state).to(args.device)
    velo[velo > 0.2] = 0
    velo_dir = math_utils.closest_one_percent(math_utils.get_velo_dir(velo), 0.01)
    indices = torch.arange(logic_state.shape[1])
    obj_datas = []
    for o_i in range(logic_state.shape[1]):
        obj_datas.append(get_symbolic_state(logic_state, velo, [o_i]).unsqueeze(1))
    obj_datas = torch.cat(obj_datas, dim=1)
    return obj_datas


def extract_boxing_kinematics(args, logic_state):
    logic_state = torch.tensor(logic_state).to(args.device)
    velo = get_state_velo(logic_state).to(args.device)
    velo[velo > 0.2] = 0
    obj_datas = []
    for o_i in range(logic_state.shape[1]):
        obj_datas.append(get_symbolic_state(logic_state, velo, [o_i]).unsqueeze(1))
    obj_datas = torch.cat(obj_datas, dim=1)
    return obj_datas


def extract_freeway_kinematics(args, logic_state):
    logic_state = torch.tensor(logic_state).to(args.device)
    velo = get_state_velo(logic_state).to(args.device)
    velo[velo > 0.2] = 0
    obj_datas = []
    for o_i in range(logic_state.shape[1]):
        obj_datas.append(get_symbolic_state(logic_state, velo, [o_i]).unsqueeze(1))
    obj_datas = torch.cat(obj_datas, dim=1)
    return obj_datas


def extract_pong_kinematics(args, logic_state):
    logic_state = torch.tensor(logic_state).to(args.device)
    velo = get_state_velo(logic_state).to(args.device)
    velo[velo > 0.2] = 0
    obj_datas = []
    for o_i in range(logic_state.shape[1]):
        obj_datas.append(get_symbolic_state(logic_state, velo, [o_i]).unsqueeze(1))
    obj_datas = torch.cat(obj_datas, dim=1)
    return obj_datas


def extract_kangaroo_kinematics(args, logic_state):
    logic_state = torch.tensor(logic_state).to(args.device)
    velo = get_state_velo(logic_state).to(args.device)
    velo[velo > 0.2] = 0
    obj_datas = []
    for o_i in range(logic_state.shape[1]):
        obj_datas.append(get_symbolic_state(logic_state, velo, [o_i]).unsqueeze(1))
    obj_datas = torch.cat(obj_datas, dim=1)
    return obj_datas


def pred_asterix_action(args, env_args, logic_state, obj_id, obj_type_model):
    if env_args.frame_i <= args.jump_frames:
        return torch.tensor(0)
    obj_id = obj_id.reshape(-1)
    if obj_id == 1:
        indices = [1, 2, 3, 4, 5, 6, 7, 8]
    elif obj_id == 2:
        indices = [9, 10, 11, 12, 13, 14, 15, 16]
    else:
        raise ValueError

    logic_state = torch.tensor(logic_state).to(obj_id.device)
    velo = get_state_velo(logic_state).to(obj_id.device)
    velo[velo > 0.2] = 0
    velo_dir = math_utils.closest_one_percent(math_utils.get_velo_dir(velo), 0.01)
    obj_data = get_symbolic_state(logic_state, velo, indices)
    action = obj_type_model(obj_data)[-1]
    action = action.argmax()
    return action


def pred_pong_action(args, env_args, logic_state, obj_id, obj_type_model):
    if env_args.frame_i <= args.jump_frames:
        return torch.tensor(0)
    obj_id = obj_id.reshape(-1)
    logic_state = torch.tensor(logic_state).to(obj_id.device)
    velo = get_state_velo(logic_state).to(obj_id.device)
    if obj_id == 1:
        i_l = 1
        i_r = 2
    elif obj_id == 2:
        i_l = 2
        i_r = 3
    else:
        raise ValueError
    pos_data = logic_state[-1, 0:1, -2:] - logic_state[-1, i_l:i_r, -2:]
    pos_dir = math_utils.closest_multiple_of_45(get_ab_dir(logic_state, 0, i_l))[-1]

    velo_data = velo[-1, i_l:i_r]
    velo_dir = math_utils.closest_one_percent(math_utils.get_velo_dir(velo), 0.01)
    velo_dir_data = velo_dir[-1, i_l:i_r]
    try:
        data = torch.cat((pos_data,
                          pos_dir.reshape(1, -1),
                          velo_data,
                          velo_dir_data.unsqueeze(1),
                          ), dim=1)
    except RuntimeError:
        print("")
    action = obj_type_model(data.reshape(-1).unsqueeze(0))
    action = action.argmax()
    return action


def pred_kangaroo_action(args, env_args, logic_state, obj_id, obj_type_model):
    if env_args.frame_i <= args.jump_frames:
        return torch.tensor(0)
    obj_id = obj_id.reshape(-1)
    if obj_id == 1:
        indices = [1]
    elif obj_id == 2:
        indices = [2, 3, 4]
    elif obj_id == 3:
        indices = [5]

    elif obj_id == 4:
        indices = [6, 7, 8, 9]
    elif obj_id == 5:
        indices = [10, 11, 12]
    elif obj_id == 6:
        indices = [13, 14, 15, 16]
    elif obj_id == 7:
        indices = [17, 18, 19]
    elif obj_id == 8:
        indices = [20, 21, 22]
    else:
        raise ValueError

    logic_state = torch.tensor(logic_state).to(obj_id.device)
    velo = get_state_velo(logic_state).to(obj_id.device)
    velo[velo > 0.2] = 0
    velo_dir = math_utils.closest_one_percent(math_utils.get_velo_dir(velo), 0.01)
    obj_data = get_symbolic_state(logic_state, velo, indices)
    action = obj_type_model(obj_data)[-1]
    action = action.argmax()
    return action


def get_symbolic_state(states, velo, indices):
    data_pos = (states[:, indices, -2:] - states[:, 0:1, -2:]).view(states.size(0), -1)
    data_pos_dirs = []
    for index in indices:
        pos_dir = math_utils.closest_multiple_of_45(get_ab_dir(states, 0, index)).view(-1, 1)
        data_pos_dirs.append(pos_dir)
    data_pos_dirs = torch.cat(data_pos_dirs, dim=1)
    data_velo = (velo[:, 0:1] - velo[:, indices]).view(velo.size(0), -1)
    velo_dir = math_utils.closest_one_percent(math_utils.get_velo_dir(data_velo.unsqueeze(1)), 0.01)
    data = torch.cat((data_pos, data_pos_dirs, data_velo, velo_dir), dim=1)
    return data


def reason_asterix(args, states, actions):
    states = torch.cat(states, dim=0)
    actions = torch.cat(actions, dim=0)
    mask_player = states[:, 0, :-6].sum(dim=-1) > 0

    x_ticks = torch.arange(-0.9, 0.9, 0.1).to(args.device)
    y_ticks = torch.arange(-0.9, 0.9, 0.1).to(args.device)
    obj_data = torch.zeros((states.shape[1], len(y_ticks), len(x_ticks)))

    pos_data = torch.zeros((states.shape[0], 2))
    for obj_i in range(1, states.shape[1]):

        mask_obj_i = states[:, obj_i, :-6].sum(dim=-1) > 0
        mask = mask_player & mask_obj_i
        op_res = states[:, 0, -2:] - states[:, obj_i, -2:]
        op_res = math_utils.closest_one_percent(op_res, unit=0.1)
        pos_data = op_res
        heat_data = torch.zeros((len(actions.unique()), len(y_ticks), len(x_ticks)))
        for a_i, action in enumerate(actions.unique()):
            action_mask = (actions == action) & mask

            values, value_counts = op_res[action_mask].unique(dim=0, return_counts=True)
            for v_i in range(len(values)):
                x_i = torch.where(values[v_i, 0] == x_ticks)[0].to(torch.int)
                y_i = torch.where(values[v_i, 1] == y_ticks)[0].to(torch.int)

                heat_data[a_i, y_i, x_i] = value_counts[v_i].to(torch.float)

        action_heat_data = heat_data.argmax(dim=0)
        obj_data[obj_i] = action_heat_data

    return


def extract_direction_obj_data(state_kinematic, relation):
    relation_degrees = relation.reshape(-1).item() * 45
    relation_dir = math_utils.closest_multiple_of_45(torch.tensor([relation_degrees]).to(torch.float)).to(
        state_kinematic.device)
    state_dirs = state_kinematic[:, 2]
    related_obj_indices = state_dirs == relation_dir
    return related_obj_indices


def asterix_obj_to_collective(obj_id):
    enemy_indices = [1, 2, 3, 4, 5, 6, 7, 8]
    consumable_indices = [9, 10, 11, 12, 13, 14, 15, 16]
    if obj_id in enemy_indices:
        return enemy_indices, 0
    elif obj_id in consumable_indices:
        return consumable_indices, 1
    else:
        return None, None
