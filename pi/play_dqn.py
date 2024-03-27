# Created by jing at 22.03.24
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
    # collect game buffer from neural agent
    if not os.path.exists(args.buffer_filename):
        teacher_agent = create_agent(args, agent_type=args.teacher_agent)
        pi.game_render.collect_data_dqn_a(teacher_agent, args, save_buffer=True)
    if args.m == "getout":
        student_agent.load_buffer(game_utils.load_buffer(args))
    else:
        student_agent.load_atari_buffer(args)

    args = game_settings.switch_hardness(args)
    if args.m=="Pong:":
        student_agent.pong_reasoner()
    if args.m=="Asterix":
        student_agent.asterix_reasoner()

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
