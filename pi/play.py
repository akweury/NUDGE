# Created by jing at 01.12.23
import torch

from src.utils_game import render_getout, render_threefish, render_loot, render_ecoinrun, render_atari
from src.agents.neural_agent import ActorCritic, NeuralPlayer
from src.agents.logic_agent import NSFR_ActorCritic, LogicPlayer
from src.agents.random_agent import RandomPlayer

from src.agents import smp_agent
from src import config
from pi.utils import log_utils, args_utils
from pi import micro_program_search
from pi import train

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


def create_agent(args, clauses):
    #### create agent
    if args.agent == "smp":
        agent = smp_agent.SymbolicMicroProgramPlayer(args, clauses)
    elif args.agent == 'random':
        agent = RandomPlayer(args)
    elif args.agent == 'human':
        agent = 'human'
    else:
        raise ValueError

    return agent


def main():

    args = args_utils.load_args(config.path_exps)
    buffer = micro_program_search.load_buffer(args)

    # behavior clauses
    behavior_clauses = micro_program_search.buffer2clauses(args, buffer)

    # weight clauses
    clause_weight_model = train.train_clause_weights(args, buffer)
    pred_actions = clause_weight_model(buffer.logic_states.unsqueeze(1)).argmax(dim=1)
    weight_clauses = micro_program_search.weights2clauses(args,buffer, pred_actions, behavior_clauses)

    # two types of clauses are both considered as game rules
    clauses = behavior_clauses # + weight_clauses

    # create a game agent
    agent = create_agent(args, clauses)

    #### Continue to render
    if args.m == 'getout':
        render_getout(agent, args)
    elif args.m == 'threefish':
        render_threefish(agent, args)
    elif args.m == 'loot':
        render_loot(agent, args)
    elif args.m == 'ecoinrun':
        render_ecoinrun(agent, args)
    elif args.m == 'atari':
        render_atari(agent, args)
    else:
        raise ValueError("Game not exist.")


if __name__ == "__main__":
    main()
