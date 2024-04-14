# Created by shaji at 14/04/2024


# Created by shaji at 13/04/2024

import time
from rtpt import RTPT

from ocatari.core import OCAtari
from nesy_pi.aitk.utils import log_utils
from tqdm import tqdm
from src import config

from nesy_pi.aitk.utils import args_utils, file_utils, game_utils, draw_utils
from nesy_pi.aitk.utils.EnvArgs import EnvArgs
from nesy_pi.aitk.utils.oc_utils import extract_logic_state_atari
from nesy_pi.aitk.utils import game_patches
from nesy_pi.aitk.utils.fol.language import Language
from nesy_pi import ilp
from nesy_pi.aitk.utils.fol import bk


def load_clauses(args):
    clause_file = args.trained_model_folder / f"learned_clauses.pkl"
    data = file_utils.load_clauses(clause_file)

    args.rule_obj_num = 10
    args.p_inv_counter = data["p_inv_counter"]
    # load logical representations
    args.clauses = data["clauses"]
    args.index_pos = config.score_example_index["pos"]
    args.index_neg = config.score_example_index["neg"]
    args.lark_path = config.path_nesy / "lark" / "exp.lark"
    args.invented_pred_num = 0
    args.batch_size = 1
    args.last_refs = []
    args.found_ns = False
    bk_preds = [bk.neural_predicate_2[bk_pred_name] for bk_pred_name in args.bk_pred_names.split(",")]
    neural_preds = file_utils.load_neural_preds(bk_preds, "bk_pred")
    args.neural_preds = [neural_pred for neural_pred in neural_preds]

    lang = Language(args, [], config.pi_type['bk'], no_init=True)
    # update language
    lang.all_clauses =[clause for clauses in  data["clauses"] for clause in clauses]
    lang.invented_preds_with_scores = []
    # update predicates
    lang.update_bk(args.neural_preds, full_bk=True)


    args.lang = lang


def init(args):
    rtpt = RTPT(name_initials='JS', experiment_name=f"{args.m}_{args.start_frame}_{args.end_frame}",
                max_iterations=args.end_frame - args.start_frame)
    # Start the RTPT tracking
    rtpt.start()

    agent = game_utils.create_agent(args, agent_type="clause")
    # Initialize environment
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()

    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    dqn_a_input_shape = env.observation_space.shape
    args.num_actions = env.action_space.n

    args.log_file = log_utils.create_log_file(args.trained_model_folder, "pi")
    agent.position_norm_factor = obs.shape[0]
    return env, env_args, agent, obs


def main():
    args = args_utils.get_args()
    load_clauses(args)
    env, env_args, agent, obs = init(args)
    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)
    for game_i in tqdm(range(args.teacher_game_nums), desc=f"ClausePlayer"):
        env_args.obs, info = env.reset()
        env_args.reset_args(game_i)
        env_args.reset_buffer_game()
        while not env_args.game_over:
            # limit frame rate
            if args.with_explain:
                current_frame_time = time.time()
                if env_args.last_frame_time + env_args.target_frame_duration > current_frame_time:
                    sl = (env_args.last_frame_time + env_args.target_frame_duration) - current_frame_time
                    time.sleep(sl)
                    continue
                env_args.last_frame_time = current_frame_time  # save frame start time for next iteration
            env_args.logic_state, env_args.state_score = extract_logic_state_atari(args, env.objects, args.game_info,
                                                                                   obs.shape[0])
            env_args.past_states.append(env_args.logic_state)
            env_args.obs = env_args.last_obs
            state = env.dqn_obs.to(args.device)
            if env_args.frame_i <= args.jump_frames:
                env_args.action = 0
            else:
                if agent.agent_type == "clause":
                    env_args.action = agent.draw_action(env_args.logic_state)
                else:
                    raise ValueError
            env_args.obs, env_args.reward, env_args.terminated, env_args.truncated, info = env.step(env_args.action)

            game_patches.atari_frame_patches(args, env_args, info)

            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, agent, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1
                env_args.update_lost_live(args.m, info["lives"])
            else:
                # record game states
                env_args.next_state, env_args.state_score = extract_logic_state_atari(args, env.objects, args.game_info,
                                                                                      obs.shape[0])
                env_args.buffer_frame("dqn_a")
            if args.with_explain:
                screen_text = (
                    f"dqn_obj ep: {env_args.game_i}, Rec: {env_args.best_score} \n "
                    f"act: {args.action_names[env_args.action]} re: {env_args.reward}")
                # Red
                env_args.obs[:10, :10] = 0
                env_args.obs[:10, :10, 0] = 255
                # Blue
                env_args.obs[:10, 10:20] = 0
                env_args.obs[:10, 10:20, 2] = 255
                draw_utils.addCustomText(env_args.obs, f"{agent.agent_type}",
                                         color=(255, 255, 255), thickness=1, font_size=0.2, pos=[2, 5])
                game_plot = draw_utils.rgb_to_bgr(env_args.obs)
                screen_plot = draw_utils.image_resize(game_plot,
                                                      int(game_plot.shape[0] * env_args.zoom_in),
                                                      int(game_plot.shape[1] * env_args.zoom_in))
                draw_utils.addText(screen_plot, screen_text,
                                   color=(255, 228, 181), thickness=2, font_size=0.6, pos="upper_right")
                video_out = draw_utils.write_video_frame(video_out, screen_plot)
            # update game args
            # update game args
            env_args.update_args()

        if args.m == "Pong":
            env_args.buffer_game(args.zero_reward, args.save_frame)
        elif args.m == "Asterix":
            env_args.buffer_game(args.zero_reward, args.save_frame)
        elif args.m == "Kangaroo":
            env_args.buffer_game(args.zero_reward, args.save_frame)
        elif args.m == "Freeway":
            env_args.buffer_game(args.zero_reward, args.save_frame)
        elif args.m == "Boxing":
            env_args.buffer_game(args.zero_reward, args.save_frame)
        else:
            raise ValueError
        env_args.game_rewards.append(env_args.rewards)

        game_utils.game_over_log(args, agent, env_args)
        env_args.reset_buffer_game()
    env.close()
    game_utils.finish_one_run(env_args, args, agent)
    if args.with_explain:
        draw_utils.release_video(video_out)


if __name__ == "__main__":
    main()
