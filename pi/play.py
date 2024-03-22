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
    # learn behaviors from data
    student_agent = create_agent(args, agent_type='smp')
    learned_negative_behaviors = [[17, 'below_of_close_to', 16, 'avoid'],
                                  [13, 'close_to', 3, 'kill'],
                                  [17, 'close_to', 2, 'avoid'],
                                  [20, 'close_to', 2, 'avoid']
                                  ]
    # collect game buffer from neural agent
    if learned_negative_behaviors is not None:
        student_agent.negative_behaviors = learned_negative_behaviors
        student_agent.update_behaviors(args=args)
        args.train_epochs = 1
        if not os.path.exists(args.buffer_filename):
            game_frame_counter = pi.game_render.train_atari_game(student_agent, args, None)
    elif not os.path.exists(args.buffer_filename):
        teacher_agent = create_agent(args, agent_type=args.teacher_agent)
        pi.game_render.render_game(teacher_agent, args, save_buffer=True)
    if args.m == "getout":
        student_agent.load_buffer(game_utils.load_buffer(args))
    else:
        student_agent.load_atari_buffer(args)

    args = game_settings.switch_hardness(args)
    pos_behs, neg_behs = student_agent.reasoning_o2o_behaviors()

    student_agent.update_behaviors(args=args)
    if args.analysis_play:
        # Test updated agent
        behavior_frames = []
        for behavior_i in range(len(neg_behs)):
            game_frame_counter = pi.game_render.train_atari_game(student_agent, args, neg_behs[behavior_i])
            behavior_frames.append(game_frame_counter)

    if render:
        # Test updated agent
        pi.game_render.render_game(student_agent, args)

    return student_agent


if __name__ == "__main__":
    main()
