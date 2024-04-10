# Created by jing at 10.04.24

import os.path
import torch
import time

from ocatari.core import OCAtari
from tqdm import tqdm
from pi.utils import args_utils
from src import config
from pi import train_utils
from pi.utils.game_utils import create_agent

from pi.utils import game_utils, reason_utils, math_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils.oc_utils import extract_logic_state_atari
from pi.utils.atari import game_patches
from pi.utils import game_utils, draw_utils


def collect_data(args, agent):
    buffer_filename = args.game_buffer_path / f"z_buffer_dqn_a_{args.teacher_game_nums}.json"
    if not os.path.exists(buffer_filename):
        game_utils.collect_data_dqn_a(agent, args, buffer_filename, save_buffer=True)

    game_buffer = game_utils.load_atari_buffer(args, buffer_filename)
    return game_buffer


def _get_kinematic_states(args, game_buffer):
    if args.m == "Freeway":
        actions = torch.cat(game_buffer["actions"], dim=0)
        states = torch.cat(game_buffer["states"], dim=0)
        kinematic_data = reason_utils.extract_freeway_kinematics(args, states)
    else:
        raise ValueError
    return kinematic_data, actions


def in_range(kinematic_data, new_param_range):
    param_num = kinematic_data.shape[1]
    data_range = torch.repeat_interleave(new_param_range.unsqueeze(0), kinematic_data.shape[0], dim=0)
    min_range_satisfaction = torch.sum((kinematic_data - data_range[:, :, 0]) >= 0, dim=1) == param_num
    max_range_satisfaction = torch.sum((data_range[:, :, 1] - kinematic_data) >= 0, dim=1) == param_num
    mask_in_range = max_range_satisfaction & min_range_satisfaction
    return mask_in_range


def same_action(actions, label_action):
    mask_action = actions == label_action
    return mask_action


def _pi_expanding(frame_i, parameter_states, actions):
    # expand

    frame_data = parameter_states[frame_i]
    param_range = torch.repeat_interleave(frame_data.unsqueeze(1), 2, dim=1)

    param_step = torch.zeros_like(param_range).to(parameter_states.device)
    param_deltas = torch.tensor([[0, 0.01], [-0.01, 0]]).to(parameter_states.device)
    mask_same_action = same_action(actions, actions[frame_i])

    scores = []
    num_cover_frames_best = 0
    for param_delta in param_deltas:
        for p_i in range(2, len(frame_data)):
            # update param range
            score_p_i_best = 0
            while True:
                param_step[p_i] = param_delta
                new_param_range = param_range + param_step
                new_param_range[new_param_range < -1] = -1
                new_param_range[new_param_range > 1] = 1
                if torch.equal(new_param_range, param_range):
                    break
                # evaluate new range and acquire score
                mask_in_range = in_range(parameter_states, new_param_range)
                num_cover_frames = (mask_in_range & mask_same_action).sum()
                score_new_range = num_cover_frames / torch.sum(mask_in_range)
                if score_new_range >= score_p_i_best and score_new_range > 0:
                    score_p_i_best = score_new_range
                    param_range = new_param_range
                    if num_cover_frames > num_cover_frames_best:
                        num_cover_frames_best = num_cover_frames
                else:
                    break
            scores.append(score_p_i_best)
    inv_score = torch.tensor(scores).max()
    return param_range, inv_score, num_cover_frames_best


def _pi_elimination(frame_i, parameter_states, actions, inv_pred, score):
    num_cover_frames = 0
    mask_param = torch.ones(inv_pred.shape[0], dtype=torch.bool)
    mask_same_action = same_action(actions, actions[frame_i])

    for p_i in range(2, inv_pred.shape[0]):
        mask_delta = torch.ones(inv_pred.shape[0], dtype=torch.bool)
        mask_delta[p_i] = False
        mask_param_new = mask_param & mask_delta

        mask_in_range = in_range(parameter_states[:, mask_param_new], inv_pred[mask_param_new])
        num_cover_frames = (mask_in_range & mask_same_action).sum()
        score_new = num_cover_frames / torch.sum(mask_in_range)
        if score_new >= score:
            mask_param = mask_param_new
            score = score_new
            if inv_pred[p_i][0] != -1 and inv_pred[p_i][1] != 1:
                print("")

    return mask_param, score, num_cover_frames


def _get_parameter_states(kinematic_data):
    player_pos = kinematic_data[:, 0:1, :2].reshape(kinematic_data.shape[0], -1)
    others_state = kinematic_data[:, 1:, 2:].reshape(kinematic_data.shape[0], -1)
    parameter_state = torch.cat((player_pos, others_state), dim=1)

    parameter_state = math_utils.closest_one_percent(parameter_state, 0.01)
    return parameter_state


def range_to_text(prefix, range_tensor):
    text = ""
    if range_tensor[0, 0] > -1:
        text += f"{prefix}_>{range_tensor[0, 0]:.2f}_"
    if range_tensor[0, 1] < 1:
        text += f"{prefix}_<{range_tensor[0, 1]:.2f}_"
    return text


def _pi_text(args, inv_pred):
    pos = (inv_pred[0, 0], inv_pred[1, 0])
    text = f"player_at({pos[0].item():.2f},{pos[1].item():.2f})_"
    trivial_pred = True
    for obj_i in range(1, len(args.row_names)):
        obj_name = f"{args.row_names[obj_i]}{obj_i}_"
        obj_index = 2 + (obj_i - 1) * 4
        x_pos = range_to_text("x", inv_pred[obj_index:obj_index + 1])
        y_pos = range_to_text("y", inv_pred[obj_index + 1:obj_index + 2])
        x_velo = range_to_text("vx", inv_pred[obj_index + 2:obj_index + 3])
        y_velo = range_to_text("vy", inv_pred[obj_index + 3:obj_index + 4])
        kinematic_text = f"{x_pos}{y_pos}{x_velo}{y_velo}"
        obj_text = ""
        if kinematic_text != "":
            obj_text = f"{obj_name}({x_pos}{y_pos}{x_velo}{y_velo})_"
            trivial_pred = False
        text += obj_text
    return text, trivial_pred


def save_pred(frame_i, inv_pred, inv_pred_file):
    data = torch.load(inv_pred_file)
    data["inv_pred"].append(inv_pred)
    data["inv_pred_frame_i"].append(frame_i)
    torch.save(data, inv_pred_file)


def create_inv_file(file_name):
    data = {"inv_pred": [], "inv_pred_frame_i": []}
    torch.save(data, file_name)


def init_pred_file(file_name, start_frame):
    if os.path.exists(file_name):
        frame_i = torch.load(file_name)["inv_pred_frame_i"]
        if len(frame_i) > 0:
            start_frame = frame_i[-1]
    else:
        create_inv_file(file_name)
    return start_frame


def main():
    args = args_utils.load_args(config.path_exps, None)
    # Initialize environment
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    args.num_actions = env.action_space.n
    dqn_a_input_shape = env.observation_space.shape
    from pi.utils import file_utils
    args.log_file = file_utils.create_log_file(args.trained_model_folder, "pi")
    # learn behaviors from data
    student_agent = create_agent(args, agent_type=args.teacher_agent)
    # collect game buffer from neural agent
    game_buffer = collect_data(args, student_agent)
    kinematic_data, actions = _get_kinematic_states(args, game_buffer)
    kinematic_data = kinematic_data.to(args.device)
    actions = actions.to(args.device)
    parameter_states = _get_parameter_states(kinematic_data)
    # train the model
    file_name = args.trained_model_folder / f"inv_pred_{args.start_frame}_{args.end_frame}.pth"
    start_frame_i = init_pred_file(file_name, args.start_frame)
    inv_preds = []
    inv_pred_scores = []
    end_frame = min(args.end_frame, len(parameter_states))
    for frame_i in tqdm(range(start_frame_i, end_frame), desc=f"Frame"):
        inv_pred, score, num_cover_frames = _pi_expanding(frame_i, parameter_states, actions)
        # mask_param, score, num_cover_frames = _pi_elimination(frame_i, parameter_states, actions, inv_pred, score)
        text, trivial_pred = _pi_text(args, inv_pred)
        file_utils.add_lines(f"(inv {frame_i}) {num_cover_frames} {args.action_names[actions[frame_i]]}:-{text}",
                             args.log_file)
        if not trivial_pred:
            inv_preds.append(inv_pred)
            inv_pred_scores.append(score)
            save_pred(frame_i, inv_pred, file_name)


if __name__ == "__main__":
    main()
