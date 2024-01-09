# Created by jing at 01.12.23
import torch

from pi.game_env import create_agent
from pi.MicroProgram import SymbolicMicroProgram
from src.agents.neural_agent import ActorCritic
from src.agents.logic_agent import NSFR_ActorCritic
from src import config

from pi.utils import args_utils
from pi import behavior, sm_program, pi_lang, game_env, game_settings


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

    for episode in range(args.episode_num):
        print(f"- Episode {episode}")
        args.episode = episode

        # prepare training data and saved as a json file
        game_env.play_games(args, agent, collect_data=True, render=False)

        # load game buffer
        buffer = game_env.load_buffer(args)

        # building symbolic microprogram
        prop_indices = game_settings.get_idx(args)
        obj_types, obj_names = game_settings.get_game_info(args)
        smp = SymbolicMicroProgram(args, buffer)

        # searching for valid behaviors
        agent_behaviors, ungrounded_behaviors = smp.programming(obj_types, prop_indices)
        # HCI: making clauses from behaviors
        clauses = pi_lang.behaviors2clauses(args, agent_behaviors)

        # update game agent
        agent.update(agent_behaviors, ungrounded_behaviors, args.obj_type_indices, clauses)

        # Test updated agent
        game_env.play_games(args, agent, collect_data=False, render=True)


if __name__ == "__main__":
    main()
