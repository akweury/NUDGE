# Created by shaji at 27/03/2024

import os.path
import torch
from ocatari.core import OCAtari
from tqdm import tqdm

from pi.utils import args_utils
from src import config
from pi import train_utils
from pi.utils.game_utils import create_agent

from pi.utils import game_utils, reason_utils
from pi.utils.EnvArgs import EnvArgs
from pi.utils.oc_utils import extract_logic_state_atari
from pi.utils.atari import game_patches
from pi import train_dqn_c



def _prepare_mlp_training_data(args, student_agent):
    if args.m == "Pong":
        actions = torch.cat(student_agent.actions, dim=0)[args.stack_num - 1:]
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_pong_kinematics(args, states)
        kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)

        args.dqn_a_avg_score = torch.sum(student_agent.buffer_win_rates > 0) / len(student_agent.buffer_win_rates)

    elif args.m == "Asterix":

        actions = torch.cat(student_agent.actions, dim=0)[args.stack_num - 1:]
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_asterix_kinematics(args, states)
        kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)

        args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)
    elif args.m == "Boxing":
        actions = torch.cat(student_agent.actions, dim=0)[args.stack_num - 1:]
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_boxing_kinematics(args, states)
        kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)

        args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)
    elif args.m == "Freeway":
        actions = torch.cat(student_agent.actions, dim=0)
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_freeway_kinematics(args, states)
        pos_data = [
            kinematic_data[:, 1:2],
            kinematic_data[:, 2:]
        ]
        args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)

    elif args.m == "Kangaroo":

        actions = torch.cat(student_agent.actions, dim=0)[args.stack_num - 1:]
        states = torch.cat(student_agent.states, dim=0)
        kinematic_data = reason_utils.extract_kangaroo_kinematics(args, states)
        kinematic_series_data = train_utils.get_stack_buffer(kinematic_data, args.stack_num)

        args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)
    else:
        raise ValueError
    return kinematic_series_data, actions

def collect_data_dqn_c(agent, args, buffer_filename, save_buffer):
    oc_name = game_utils.get_ocname(args.m)
    # load mlp_a
    obj_type_num = len(args.game_info["obj_info"]) - 1
    mlp_a = train_utils.load_mlp_a(args, args.trained_model_folder, obj_type_num, args.m)

    env = OCAtari(oc_name, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    agent.position_norm_factor = obs.shape[0]
    for game_i in tqdm(range(args.teacher_game_nums), desc=f"Collecting GameBuffer by {agent.agent_type}"):
        env_args.obs, info = env.reset()
        env_args.reset_args(game_i)
        env_args.reset_buffer_game()
        while not env_args.game_over:
            # predict object type
            collective_pred = agent.select_action(env.dqn_obs.to(env_args.device))
            env_args.logic_state, _ = extract_logic_state_atari(args, env.objects, args.game_info, obs.shape[0])
            env_args.past_states.append(env_args.logic_state)


            if env_args.frame_i <= args.jump_frames:
                action = torch.tensor([[0]]).to(args.device)
                collective_pred = torch.tensor([[0]]).to(args.device)
            else:
                action = train_dqn_c._reason_action(args, env_args, collective_pred + 1, mlp_a)

            # if args.m == "Asterix":
            #     action = reason_utils.pred_asterix_action(args, env_args, env_args.past_states, collective_pred + 1,
            #                                               mlp_a[collective_pred]).to(torch.int64).reshape(1)
            # elif args.m == "Pong":
            #     action = reason_utils.pred_pong_action(args, env_args, env_args.past_states, collective_pred + 1,
            #                                            mlp_a[collective_pred]).to(torch.int64).reshape(1)
            # elif args.m == "Kangaroo":
            #     action = reason_utils.pred_kangaroo_action(args, env_args, env_args.past_states, collective_pred + 1,
            #                                                mlp_a[collective_pred]).to(torch.int64).reshape(1)
            # else:
            #     raise ValueError

            state = env.dqn_obs.to(args.device)
            env_args.obs, env_args.reward, env_args.terminated, env_args.truncated, info = env.step(action)
            game_patches.atari_frame_patches(args, env_args, info)
            if info["lives"] < env_args.current_lives or env_args.truncated or env_args.terminated:
                game_patches.atari_patches(args, agent, env_args, info)
                env_args.frame_i = len(env_args.logic_states) - 1
                env_args.update_lost_live(args.m, info["lives"])
            else:
                # record game states
                env_args.next_state, env_args.state_score = extract_logic_state_atari(args, env.objects, args.game_info,
                                                                                      obs.shape[0])
                env_args.action = action
                env_args.collective = collective_pred.reshape(-1).item()
                env_args.buffer_frame("dqn_c")
            env_args.frame_i += 1

            env_args.reward = torch.tensor(env_args.reward).reshape(1).to(args.device)
            next_state = env.dqn_obs.to(args.device) if not env_args.terminated else None
            # Store the transition in memory
            agent.memory.push(state, collective_pred, next_state, env_args.reward, env_args.terminated)

        if args.m == "Pong":
            if sum(env_args.rewards) > 0:
                env_args.buffer_game(args.zero_reward, args.save_frame)
                game_utils.game_over_log(args, agent, env_args)
        elif args.m == "Asterix":
            env_args.buffer_game(args.zero_reward, args.save_frame)
            game_utils.game_over_log(args, agent, env_args)
        elif args.m == "Kangaroo":
            env_args.buffer_game(args.zero_reward, args.save_frame)
            game_utils.game_over_log(args, agent, env_args)
        else:
            raise ValueError

        env_args.reset_buffer_game()

    env.close()
    game_utils.finish_one_run(env_args, args, agent)
    if save_buffer:
        game_utils.save_game_buffer(args, env_args, buffer_filename)


def train_mlp_c():
    # game buffer
    args = args_utils.load_args(config.path_exps, None)
    # train mlp-t
    env = OCAtari(args.m, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    num_actions = env.action_space.n
    dqn_t_input_shape = env.observation_space.shape
    obj_type_num = len(args.game_info["obj_info"]) - 1
    student_agent = create_agent(args, agent_type='smp')
    # collect game buffer from neural agent
    buffer_filename = args.game_buffer_path / f"z_buffer_dqn_c_{args.teacher_game_nums}.json"
    if not os.path.exists(buffer_filename):
        # load dqn-t agent
        dqn_c_agent = train_utils.DQNAgent(args, dqn_t_input_shape, obj_type_num)
        dqn_c_agent.agent_type = "DQN-C"
        train_utils.load_dqn_c(args, dqn_c_agent, args.trained_model_folder)
        collect_data_dqn_c(dqn_c_agent, args, buffer_filename, save_buffer=True)
    student_agent.load_atari_buffer(args, buffer_filename)

    pos_data, actions = _prepare_mlp_training_data(args, student_agent)

    # args.dqn_a_avg_score = torch.mean(student_agent.buffer_win_rates)

    # if args.m == "Pong":
    #     pos_data, actions = student_agent.pong_reasoner()
    # if args.m == "Asterix":
    #     pos_data, actions = student_agent.asterix_reasoner()
    # if args.m == "Kangaroo":
    #     pos_data, actions = student_agent.kangaroo_reasoner()
    # convert to symbolic input
    # input_tensor = torch.cat(pos_data, dim=1).to(args.device)
    input_tensor = pos_data.view(pos_data.size(0), -1)
    target_tensor = actions.to(args.device)

    act_pred_model_file = args.trained_model_folder / f"{args.m}_mlp_c.pth.tar"

    if not os.path.exists(act_pred_model_file):
        target_pred_model = train_utils.train_nn(args, num_actions, input_tensor, target_tensor,
                                                 f"mlp_c")
        state = {'model': target_pred_model}
        torch.save(state, act_pred_model_file)
    else:
        target_pred_model = torch.load(act_pred_model_file)["model"]

    return args.dqn_a_avg_score


if __name__ == "__main__":
    train_mlp_c()
