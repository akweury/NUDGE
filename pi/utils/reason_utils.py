# Created by jing at 01.03.24
import torch
from itertools import combinations
from pi.utils import file_utils, math_utils, draw_utils


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

    for i in range(1, len(data) - 1):
        if data[i - 1] < data[i] > data[i + 1]:
            local_max_indices.append(i)
        elif data[i - 1] > data[i] < data[i + 1]:
            local_min_indices.append(i)

    return torch.tensor(local_min_indices), torch.tensor(local_max_indices)


def get_key_frame_batch(data_batch):
    # Find local maximum and minimum indices
    local_min_frames = []
    local_max_frames = []
    for data in data_batch:
        local_min, local_max = get_key_frame(data)
        local_min_frames.append(local_min.tolist())
        local_max_frames.append(local_max.tolist())
    key_frames = {'local_min': local_min_frames, 'local_max': local_max_frames}
    return key_frames


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
    prop_combs = math_utils.all_subsets(list(range(2)))
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


def text_from_tensor(o2o_data, state_tensors):
    explain_text = ""
    prop_explain = {0: 'dx', 1: 'dy', 2: 'va', 3: 'vb', 4: 'dir_ab'}
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


def reason_o2o_states(args, data):
    behavior_data = {'behavior_data': [], 'action_data': []}
    states = torch.cat([data['win_states'], data['lost_states']], dim=0)
    actions = torch.cat([data['win_actions'], data['lost_actions']], dim=0)
    rewards = torch.cat([data['win_rewards'], data['lost_rewards']], dim=0)
    if len(data['win_states']) > 0:
        obj_a_id = 0
        obj_b_id = 1
        state_tensors = state2analysis_tensor(states, obj_a_id, obj_b_id)
        # for s_ in range(state_tensors.shape[1]):
        #     state_tensors[:, s_] = math_utils.smooth_action(state_tensors[:, s_])
        key_frames = get_key_frame_batch(state_tensors.permute(1, 0))
        stat_pf_data = stat_pf_behaviors(state_tensors, key_frames)
        # stat_pos_data = stat_reward_behaviors(state_tensors, key_frames, actions, rewards)

        state_pos_tensors = state2pos_tensor(states, obj_a_id, obj_b_id)
        stat_act_data = stat_o2o_action(args, state_pos_tensors, actions)

        behavior_data['action_data'] = stat_act_data
        behavior_data['behavior_data'] += stat_pf_data

        # draw_utils.plot_heat_map(stat_pos_data.permute(1, 0)[:, :150], path=args.check_point_path / "o2o",
        #                          name=f"win_states_{args.m}", figsize=(40, 3), key_col=75,
        #                          row_names=['dx', 'dy', 'vdir', 'act', 'd_rew', 'rew'])
        key_cols = []
        neg_cols = []
        key_pos_x_pos = []
        key_pos_y_pos = []
        key_pos_va_pos = []
        key_pos_vb_pos = []
        key_dir_pos = []
        key_pos_x_count = []
        key_pos_y_count = []
        key_pos_va_count = []
        key_pos_vb_count = []
        key_pos_dir_count = []

        neg_x = []
        neg_y = []
        neg_va = []
        neg_vb = []
        neg_dir_ab = []
        neg_x_count = []
        neg_y_count = []
        neg_va_count = []
        neg_vb_count = []
        neg_dir_count = []

        for data in stat_pf_data:
            if data[3] == 'local_min':
                neg_cols += data[2]
                if 0 in data[0]:
                    neg_x += data[1]
                    neg_x_count += [len(data[2])] * len(data[1])

                if 1 in data[0]:
                    neg_y += data[1]
                    neg_y_count += [len(data[2])] * len(data[1])

                if 2 in data[0]:
                    neg_va += data[1]
                    neg_va_count += [len(data[2])] * len(data[1])

                if 3 in data[0]:
                    neg_vb += data[1]
                    neg_vb_count += [len(data[2])] * len(data[1])

                if 4 in data[0]:
                    neg_dir_ab += data[1]
                    neg_dir_count += [len(data[2])] * len(data[1])

            else:
                key_cols += data[2]
                if 0 in data[0]:
                    key_pos_x_pos += data[1]
                    key_pos_x_count += [len(data[2])] * len(data[1])
                if 1 in data[0]:
                    key_pos_y_pos += data[1]
                    key_pos_y_count += [len(data[2])] * len(data[1])
                if 2 in data[0]:
                    key_pos_va_pos += data[1]
                    key_pos_va_count += [len(data[2])] * len(data[1])
                if 3 in data[0]:
                    key_pos_vb_pos += data[1]
                    key_pos_vb_count += [len(data[2])] * len(data[1])
                if 4 in data[0]:
                    key_dir_pos += data[1]
                    key_pos_dir_count += [len(data[2])] * len(data[1])

        key_cols = sorted(list(set(key_cols)))
        key_cols = [k for k in key_cols if k < 100]

        neg_cols = sorted((list(set(neg_cols))))
        neg_cols = [k for k in neg_cols if k < 100]

        draw_utils.plot_compare_line_chart(state_tensors.permute(1, 0)[:, :100].tolist(), path=args.check_point_path / "o2o",
                                           name=f"win_states_{args.m}", figsize=(30, 20),
                                           key_cols=key_cols,
                                           key_rows=[key_pos_x_pos, key_pos_y_pos, key_pos_va_pos, key_pos_vb_pos,
                                                     key_dir_pos],
                                           neg_rows=[neg_x, neg_y, neg_va, neg_vb, neg_dir_ab],
                                           neg_cols=neg_cols,
                                           key_name=[key_pos_x_count, key_pos_y_count, key_pos_va_count,
                                                     key_pos_vb_count, key_pos_dir_count],
                                           neg_name=[neg_x_count, neg_y_count, neg_va_count, neg_vb_count,
                                                     neg_dir_count],
                                           pos_color="orange",
                                           neg_color="blue",
                                           row_names=['dx_0_1', 'dy_0_1', 'vadir', 'vbdir', 'dir_ab'])
    file_utils.save_json(args.o2o_data_file, behavior_data)
    return behavior_data
