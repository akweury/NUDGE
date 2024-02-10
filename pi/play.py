# Created by jing at 01.12.23
import torch

from pi.game_env import create_agent
from pi.MicroProgram import SymbolicMicroProgram
from pi import game_env, game_settings
from pi.utils import game_utils, args_utils
from pi import pi_lang
from src.agents.neural_agent import ActorCritic
from src.agents.logic_agent import NSFR_ActorCritic
from src import config


def load_model(args, set_eval=True):
    if args.agent in ['random', 'human']:
        return None

    if args.agent == "logic":
        model_name = str(config.path_model / args.m / args.agent / "beam_search_top1.pth")
        with open(model_name, "rb") as f:
            model = NSFR_ActorCritic(args).to(args.device)
            model.load_state_dict(state_dict=torch.load(f, map_location=args.device))
    elif args.agent == "ppo":
        model_name = str(config.path_model / args.m / args.agent / "ppo_.pth")
        with open(model_name, "rb") as f:
            model = ActorCritic(args).to(args.device)
            model.load_state_dict(state_dict=torch.load(f, map_location=args.device))
    else:
        raise ValueError

    model = model.actor
    # model.as_dict = True

    if args.agent == 'logic':
        model.print_program()

    # if set_eval:
    #     model = model.eval()

    return model


def main(render=True, m=None):
    # load arguments
    args = args_utils.load_args(config.path_exps, m)
    teacher_agent = create_agent(args, agent_type=args.teacher_agent)
    # if render:
    #     game_env.render_game(teacher_agent, args)

    game_env.collect_data_game(teacher_agent, args)
    # learn behaviors from data
    agent = create_agent(args, agent_type='smp')
    # args.agent_type = 'smp'
    # building symbolic microprogram
    agent.prop_indices = game_settings.get_idx(args)
    agent.load_buffer(game_utils.load_buffer(args))
    pf_behaviors = agent.reasoning_pf_behaviors()
    def_behaviors,_  = agent.reasoning_def_behaviors()
    # att_behaviors = agent.reasoning_att_behaviors()
    agent.update_behaviors(pf_behaviors, def_behaviors, None, args)
    # args.m = "getoutplus"
    # args.obj_info = config.obj_info_getoutplus
    if render:
        # Test updated agent
        game_env.render_game(agent, args)

    return agent


if __name__ == "__main__":
    main()
