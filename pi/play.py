# Created by jing at 01.12.23
import os.path

import pi.game_render
from pi.utils.game_utils import create_agent

from pi import game_buffer, game_settings
from pi.utils import game_utils, args_utils
from src import config

num_cores = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(1)


def main(render=True, m=None):
    # load arguments
    args = args_utils.load_args(config.path_exps, m)
    # collect game buffer from neural agent
    if not os.path.exists(args.buffer_filename):
        teacher_agent = create_agent(args, agent_type=args.teacher_agent)
        pi.game_render.render_game(teacher_agent, args, save_buffer=True)
    # learn behaviors from data
    student_agent = create_agent(args, agent_type='smp')
    student_agent.prop_indices = game_settings.get_idx(args)

    if args.m == "getout":
        student_agent.load_buffer(game_utils.load_buffer(args))
    else:
        student_agent.load_atari_buffer(args)
    args = game_settings.switch_hardness(args)
    pf_behaviors = None
    def_behaviors = None
    att_behaviors = None
    att_skill = None
    o2o_behaviors = None
    o2o_data = student_agent.reasoning_o2o_behaviors()
    student_agent.train_state_estimator()

    # att_skill = agent.reasoning_att_skills()
    # att_behaviors = agent.reasoning_att_behaviors()

    # pf_behaviors = agent.reasoning_path_behaviors()
    # def_behaviors = agent.reasoning_def_behaviors()
    student_agent.update_behaviors(pf_behaviors=pf_behaviors, def_behaviors=def_behaviors,
                                   att_behaviors=att_behaviors,
                                   skill_att_behavior=att_skill,
                                   o2o_behaviors=o2o_behaviors,
                                   args=args)
    if args.analysis_play:
        if not os.path.exists(args.o2o_weight_file):
            # Test updated agent
            teacher_agent = create_agent(args, agent_type=args.teacher_agent)
            pi.game_render.train_atari_game(teacher_agent, student_agent, args, o2o_data)
    if render:
        # Test updated agent
        pi.game_render.render_game(student_agent, args)

    return student_agent


if __name__ == "__main__":
    main()
