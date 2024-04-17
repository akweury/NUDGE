# Created by jing at 15.04.24


import time
from rtpt import RTPT
import numpy as np
from ocatari.core import OCAtari
from nesy_pi.aitk.utils import log_utils
from tqdm import tqdm
from src import config
import torch
from nesy_pi.aitk.utils import args_utils, file_utils, game_utils, draw_utils
from nesy_pi.aitk.utils.EnvArgs import EnvArgs
from nesy_pi.aitk.utils.oc_utils import extract_logic_state_atari
from nesy_pi.aitk.utils import game_patches
from nesy_pi.aitk.utils.fol.language import Language
from nesy_pi import ilp
from nesy_pi.aitk.utils.fol import bk


def load_clauses(args):
    clause_file = args.trained_model_folder / args.learned_clause_file
    data = file_utils.load_clauses(clause_file)

    args.rule_obj_num = 10
    args.p_inv_counter = data["p_inv_counter"]
    # load logical representations
    clauses_with_scores = [cs for acs in data["clauses"] for cs in acs]
    args.clause_scores = torch.cat([cs[1].unsqueeze(0) for cs in clauses_with_scores], dim=0).to(args.device)
    args.clauses = [cs[0] for cs in clauses_with_scores]
    args.index_pos = config.score_example_index["pos"]
    args.index_neg = config.score_example_index["neg"]
    args.lark_path = config.path_nesy / "lark" / "exp.lark"
    args.invented_pred_num = data["p_inv_counter"]
    args.batch_size = 1
    args.last_refs = []
    args.found_ns = False
    bk_preds = [bk.neural_predicate_2[bk_pred_name] for bk_pred_name in args.bk_pred_names.split(",")]
    neural_preds = file_utils.load_neural_preds(bk_preds, "bk_pred")
    args.neural_preds = [neural_pred for neural_pred in neural_preds]

    lang = Language(args, [], config.pi_type['bk'], no_init=True)
    # update language
    lang.all_clauses = args.clauses
    lang.invented_preds_with_scores = []
    lang.all_pi_clauses = data["all_pi_clauses"]
    lang.all_invented_preds = data["all_invented_preds"]
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


from src.agents.utils_getout import extract_logic_state_getout
from src.environments.getout.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from src.environments.getout.getout.getout.getout import Getout


def _render(args, agent, env_args, video_out, agent_type):
    # render the game

    screen_text = (
        f"{agent.agent_type} ep: {env_args.game_i}, Rec: {env_args.best_score} \n "
        f"act: {args.action_names[env_args.action]} re: {env_args.reward}")
    # env_args.logic_state = agent.now_state
    video_out, _ = game_utils.plot_game_frame(agent_type, env_args, video_out, env_args.obs, screen_text)


def main():
    args = args_utils.get_args()
    load_clauses(args)
    # env, env_args, agent, obs = init(args)
    agent = game_utils.create_agent(args, agent_type="clause")
    teacher_agent = game_utils.create_agent(args, agent_type="ppo")
    def create_getout_instance(args, seed=None):
        if args.hardness == 1:
            enemies = True
        else:
            enemies = False
        # level_generator = DummyGenerator()
        getout = Getout()
        level_generator = ParameterizedLevelGenerator(enemies=enemies)
        level_generator.generate(getout, seed=seed)
        getout.render()

        return getout

    env = create_getout_instance(args)
    env_args = EnvArgs(agent=agent, args=args, window_size=[env.camera.height, env.camera.width], fps=60)
    agent.position_norm_factor = env.camera.height
    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)
    for game_i in tqdm(range(env_args.game_num), desc=f"Agent  {agent.agent_type}"):
        env = create_getout_instance(args)
        env_args.reset_args(game_i)
        env_args.reset_buffer_game()
        while not env_args.game_over:
            if env_args.frame_i > 300:
                break
            # limit frame rate
            if args.with_explain:
                current_frame_time = time.time()
                if env_args.last_frame_time + env_args.target_frame_duration > current_frame_time:
                    sl = (env_args.last_frame_time + env_args.target_frame_duration) - current_frame_time
                    time.sleep(sl)
                    continue
                env_args.last_frame_time = current_frame_time  # save frame start time for next iteration

            # agent predict an action
            env_args.logic_state = extract_logic_state_getout(env, args).squeeze().tolist()

            env_args.obs = env_args.last_obs
            teacher_action = teacher_agent.reasoning_act(env)
            env_args.action = agent.learn_action_weights(env_args.logic_state, teacher_action)
            try:
                env_args.reward = env.step(env_args.action + 1)
            except KeyError:
                env.level.terminated = True
                env.level.lost = True
                env_args.terminated = True
                env_args.truncated = True
                break
            env_args.last_obs = np.array(env.camera.screen.convert("RGB"))
            if env.level.terminated:
                env_args.frame_i = len(env_args.logic_states) - 1

                # revise the game rules
                if agent.agent_type == "smp" and len(
                        env_args.logic_states) > 2 and game_i % args.reasoning_gap == 0 and args.revise:
                    agent.revise_loss(args, env_args)
                    if args.with_explain:
                        game_utils.revise_loss_log(env_args, agent, video_out)
                env_args.game_over = True

            else:
                if args.with_explain:
                    _render(args, agent, env_args, video_out, "")

                    game_utils.frame_log(agent, env_args)
            # update game args
            env_args.update_args()
        if not env.level.lost:
            env_args.win_rate[game_i] = env_args.win_rate[game_i - 1] + 1
        else:
            env_args.win_rate[game_i] = env_args.win_rate[game_i - 1]
        env_args.state_score = env_args.win_rate[game_i - 1]

        env_args.reset_buffer_game()
        # game_utils.game_over_log(args, agent, env_args)

    game_utils.finish_one_run(env_args, args, agent)
    if args.with_explain:
        draw_utils.release_video(video_out)


if __name__ == "__main__":
    main()
