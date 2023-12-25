# Created by jing at 01.12.23
import torch

import pi.sm_program
from src.utils_game import render_getout, render_threefish, render_loot, render_ecoinrun, render_atari
from src.agents.neural_agent import ActorCritic, NeuralPlayer
from src.agents.logic_agent import NSFR_ActorCritic, LogicPlayer
from src.agents.random_agent import RandomPlayer
from src.agents import smp_agent
from src import config

from pi.utils import log_utils, args_utils
from pi import behavior, sm_program, pi_lang



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


def create_agent(args, clauses, smps):
    #### create agent
    if args.agent == "smp":
        agent = smp_agent.SymbolicMicroProgramPlayer(args, clauses, smps)
    elif args.agent == 'random':
        agent = RandomPlayer(args)
    elif args.agent == 'human':
        agent = 'human'
    else:
        raise ValueError

    return agent


def main():
    args = args_utils.load_args(config.path_exps)
    buffer = behavior.load_buffer(args)

    # train an action predicting model
    # action_imitation_model = train.train_clause_weights(args, buffer)
    # pred_actions = action_imitation_model(buffer.logic_states.unsqueeze(1)).argmax(dim=1)

    # observe behaviors from buffer
    agent_behaviors = behavior.buffer2behaviors(args, buffer)

    # HCI: making clauses from behaviors
    clauses = pi_lang.behaviors2clauses(args, agent_behaviors)

    # making behavior symbolic microprogram
    behavior_smps = sm_program.behavior2smps(args,buffer, agent_behaviors)

    # create a game agent
    agent = create_agent(args, clauses, behavior_smps)

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
