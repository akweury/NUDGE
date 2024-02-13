# Created by jing at 01.12.23
import pi.game_render
from pi.utils.game_utils import create_agent

from pi import game_buffer, game_settings
from pi.utils import game_utils, args_utils
from src import config


def main(render=True, m=None):
    # load arguments
    args = args_utils.load_args(config.path_exps, m)
    teacher_agent = create_agent(args, agent_type=args.teacher_agent)
    # if render:
    #     pi.game_render.render_game(teacher_agent, args)
    game_buffer.collect_data_game(teacher_agent, args)
    # learn behaviors from data
    agent = create_agent(args, agent_type='smp')
    agent.prop_indices = game_settings.get_idx(args)

    if args.m == "getout":
        agent.load_buffer(game_utils.load_buffer(args))
    else:
        agent.load_atari_buffer(args)
    args = game_settings.switch_hardness(args)
    pf_behaviors = agent.reasoning_pf_behaviors()
    def_behaviors, _ = agent.reasoning_def_behaviors()
    # att_behaviors = agent.reasoning_att_behaviors()
    agent.update_behaviors(pf_behaviors=pf_behaviors, def_behaviors=def_behaviors, att_behaviors=None, args=args)

    if render:
        # Test updated agent
        pi.game_render.render_game(agent, args)

    return agent


if __name__ == "__main__":
    main()
