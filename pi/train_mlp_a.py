# Created by shaji at 26/03/2024

import os.path
import torch
from ocatari.core import OCAtari

from pi.utils import args_utils
from src import config
from pi import train_utils
from pi.utils.game_utils import create_agent
from pi.game_render import collect_full_data


def train_mlp_a():
    args = args_utils.load_args(config.path_exps, None)
    # learn behaviors from data
    student_agent = create_agent(args, agent_type='smp')
    # collect game buffer from neural agent
    if not os.path.exists(args.buffer_filename):
        teacher_agent = create_agent(args, agent_type=args.teacher_agent)
        collect_full_data(teacher_agent, args, save_buffer=True)
    student_agent.load_atari_buffer(args)

    if args.m == "Pong":
        pos_data, actions = student_agent.pong_reasoner()
    if args.m == "Asterix":
        pos_data, actions = student_agent.asterix_reasoner()
    if args.m == "Kangaroo":
        pos_data, actions = student_agent.kangaroo_reasoner()
    # Initialize environment
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    num_actions = env.action_space.n

    obj_type_models = []
    for obj_type in range(len(pos_data)):
        input_tensor = pos_data[obj_type].to(args.device)
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        target_tensor = actions.to(args.device)

        act_pred_model_file = args.trained_model_folder / f"{args.m}_{obj_type}.pth.tar"

        if not os.path.exists(act_pred_model_file):
            action_pred_model = train_utils.train_nn(args, num_actions, input_tensor, target_tensor,
                                                     f"mlp_a_{obj_type}")
            state = {'model': action_pred_model}
            torch.save(state, act_pred_model_file)
        else:
            action_pred_model = torch.load(act_pred_model_file)["model"]
        obj_type_models.append(action_pred_model)


if __name__ == "__main__":
    train_mlp_a()
