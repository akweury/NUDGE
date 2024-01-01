# Created by jing at 01.12.23
import torch

from pi.game_env import create_agent
from src.utils_game import render_getout, render_threefish, render_loot, render_ecoinrun, render_atari
from src.agents.neural_agent import ActorCritic
from src.agents.logic_agent import NSFR_ActorCritic
from src import config

from pi.utils import args_utils
from pi import behavior, sm_program, pi_lang, game_env


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


def main():
    # load arguments
    args = args_utils.load_args(config.path_exps)

    # create a game agent
    agent = create_agent(args)

    # prepare training data and saved as a json file
    game_env.play_games_and_collect_data(args, agent)

    # load game buffer
    buffer = game_env.load_buffer(args)

    # observe behaviors from buffer
    agent_behaviors = behavior.buffer2behaviors(args, buffer)

    # HCI: making clauses from behaviors
    clauses = pi_lang.behaviors2clauses(args, agent_behaviors)

    # making behavior symbolic microprogram
    behavior_smps = sm_program.behavior2smps(args, buffer, agent_behaviors)

    # update game agent
    agent.update(behavior_smps)

    # Test updated agent
    game_env.play_games_and_render(args, agent)


if __name__ == "__main__":
    main()
