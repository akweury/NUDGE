# Created by jing at 30.04.24

import argparse
import csv
import sys
import time

import gym
import numpy as np
import wandb

sys.path.insert(0, '../')

from ocatari.core import OCAtari
from rtpt import RTPT
from tqdm import tqdm

from src.agents.logic_agent import LogicPPO
from src.agents.neural_agent import NeuralPPO
from src.config import *
from src.environments.procgen.procgen import ProcgenGym3Env
# from make_graph import plot_weights
from src.utils import env_step, initialize_game, make_deterministic
from torch.utils.tensorboard import SummaryWriter
import datetime

from nesy_pi import getout_utils
from nesy_pi.aitk.utils import args_utils, game_utils
from src.agents import utils_getout
from nesy_pi.aitk.utils.EnvArgs import EnvArgs


def main():
    ################### args definition ###################
    parser = argparse.ArgumentParser()
    parser.add_argument("--player_num", type=int, default=1, help="Number of Players in the game.")
    parser.add_argument("--device", help="cpu or cuda", default="cpu", type=str)
    parser.add_argument('-d', '--dataset', required=False, help='the dataset to load if scoring', dest='d')
    parser.add_argument("--cim-step", type=int, default=5,
                        help="The steps of clause infer module.")
    parser.add_argument('--gamma', default=0.001, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--with_pi", action="store_true", help="Generate Clause with predicate invention.")
    parser.add_argument("--with_explain", help="explain the game", action="store_true", default=False)
    parser.add_argument("--learned_clause_file", type=str)
    parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                        required=False, action="store", dest="seed", type=int, default=0)
    parser.add_argument("-alg", "--algorithm", help="algorithm that to use",
                        action="store", dest="alg", required=True,
                        choices=['ppo', 'logic'])
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m",
                        choices=['getout', 'threefish', 'loot', 'atari'])
    parser.add_argument("-env", "--environment", help="environment of game to use",
                        required=True, action="store", dest="env",
                        choices=['getout', 'threefish', 'loot', 'Freeway', 'kangaroo', 'Asterix', 'loothard'])
    parser.add_argument("-r", "--rules", dest="rules", default=None, required=False)
    parser.add_argument('-p', '--plot', help="plot the image of weights", type=bool, default=False, dest='plot')
    parser.add_argument('-re', '--recovery', action="store_true", help='recover from crash', default=False,
                        dest='recover')
    # arg = ['-alg', 'logic', '-m', 'threefish.json', '-env', 'threefish.json', '-p', 'True', '-r', 'threefish_human_assisted']
    args = parser.parse_args()
    if args.device != "cpu":
        args.device = int(args.device)

    args_file = path_args / f"{args.env}.json"
    args_utils.load_args_from_file(str(args_file), args)
    args.trained_model_folder = path_check_point / f"{args.env}" / "trained_models"
    args.rule_obj_num = 10

    #####################################################
    # load environment
    print("training environment name : " + args.env.capitalize())
    make_deterministic(args.seed)

    #####################################################
    # config setting
    if args.alg == 'ppo':
        update_timestep = max_ep_len * 4
    elif args.alg == 'logic' and args.m == 'atari':
        # a large num causes out of memory
        update_timestep = 100
        # print("PUT BACK 20 ! ")
        # update_timestep = 7
        # max_ep_len = 100
    else:
        update_timestep = max_ep_len * 2

    if args.m == 'loot' and args.alg == 'ppo':
        max_training_timesteps = 5000000
    else:
        max_training_timesteps = 1000000
    #####################################################

    if args.m == "getout":
        env = getout_utils.create_getout_instance(args)
    elif args.m == "threefish" or args.m == 'loot':
        env = ProcgenGym3Env(num=1, env_name=args.env, render_mode=None)
        reward, obs, done = env.observe()

    elif args.m == "atari":
        env = OCAtari(env_name=args.env.capitalize(), mode="revised", render_mode="rgb_array")
        obs, info = env.reset()

        # env = OCAtari(env_name='Freeway', mode="revised")

    #####################################################
    # config = {
    #     "seed": args.seed,
    #     "learning_rate_actor": lr_actor,
    #     "learning_rate_critic": lr_critic,
    #     "epochs": K_epochs,
    #     "gamma": gamma,
    #     "eps_clip": eps_clip,
    #     "max_steps": max_training_timesteps,
    #     "eps start": 1.0,
    #     "eps end": 0.02,
    #     "max_ep_len": max_ep_len,
    #     "update_freq": max_ep_len * 2,
    #     "save_freq": max_ep_len * 50,
    # }
    # if args.rules is not None:
    #     runs_name = str(args.rules) + '_seed_' + str(args.seed)
    # else:
    #     runs_name = str(args.m) + '_' + args.alg + '_seed_' + str(args.seed)

    # wandb.init(project="GETOUT-BS", entity="nyrus", config=config, name=runs_name)
    # wandb.init(project="LOOT", entity="nyrus", config=config, name=runs_name)
    # wandb.init(project="THREEFISH", entity="nyrus", config=config, name=runs_name)

    ################### checkpointing ###################

    if not os.path.exists(str(path_check_point)):
        os.makedirs(str(path_check_point))

    if args.rules is not None:
        directory = args.trained_model_folder / f"rho_{args.rho_num}_phi_{args.phi_num}_step_{args.max_step}" / str(
            args.seed)
    else:
        directory = path_check_point / args.m / args.alg / args.env / str(args.seed)
    if not os.path.exists(str(directory)):
        os.makedirs(str(directory))

    # if not args.recover:

    checkpoint_path = directory / "{}_{}.pth".format(args.env, 0)

    print("save checkpoint path : " + str(checkpoint_path))

    #####################################################

    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    # print("state space dimension : ", state_dim)
    # print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")

    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################
    #
    # initialize agent
    if args.alg == "ppo":
        agent = NeuralPPO(lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args)
    elif args.alg == "logic":
        agent = LogicPPO(lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args)
        print('Candidate Clauses:')
        for clause in agent.policy.actor.clauses:
            print(clause)

    time_step = 0
    i_episode = 0

    if args.recover:
        if args.alg == 'logic':
            step_list, reward_list, weights_list = agent.load(directory)
            time_step = max(step_list)[0]
        else:
            step_list, reward_list = agent.load(directory)
            time_step = max(step_list)[0]
    else:
        step_list = []
        reward_list = []
        weights_list = []
    from nesy_pi.aitk.utils import draw_utils
    if len(reward_list) > 1:
        draw_utils.plot_line_chart(torch.tensor(reward_list).permute(1, 0),
                                   args.trained_model_folder,
                                   ["reward"], x=torch.tensor(step_list).squeeze(), cla_leg=True,
                                   title=f"{args.env}_reward_{args.seed}.png"
                                   )

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"{args.env}",

        # track hyperparameters and run metadata
        config={
            "architecture": "PI",
            "dataset": f"{args.env}",
            "epochs": max_training_timesteps // print_freq,
            "rho_num": args.rho_num,
            "phi_num": args.phi_num,
            "clauses": args.clauses,
            "clause_num": len(args.clauses),
            "learned_clause_file": args.learned_clause_file
        }
    )

    # track total training time
    start_time = time.time()
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    if not os.path.exists(str(path_image)):
        os.makedirs(str(path_image))

    if args.rules:
        image_directory = path_image / args.m / args.env / args.rules / str(args.seed)
    else:
        image_directory = path_image / args.m / args.env / str(args.seed)
    if not os.path.exists(str(image_directory)):
        os.makedirs(str(image_directory))
    if args.m == "getout":
        env_args = EnvArgs(agent=agent, args=args, window_size=[env.camera.height,
                                                                env.camera.width], fps=60)

    elif args.m == "loot" or args.m=="threefish":
        env_args = EnvArgs(agent=agent, args=args, window_size=None, fps=60)
    else:
        env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    if args.with_explain:
        video_out = game_utils.get_game_viewer(env_args)
    # if args.plot:
    #     if args.alg == 'logic':
    #         plot_weights(agent.get_weights(), image_directory)

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    rtpt = RTPT(name_initials='JS', experiment_name=f'{args.env}phi{args.phi_num}',
                max_iterations=max_training_timesteps - time_step)

    # Start the RTPT tracking
    folder_name = f"{args.m}_{args.env}_{args.alg}_rho_{args.rho_num}_phi_{args.phi_num}_s{args.seed}_pi_{args.with_pi}"
    folder_name += datetime.datetime.now().strftime("%m%d-%H_%M")
    writer = SummaryWriter(str(args.trained_model_folder / folder_name))
    rtpt.start()
    # training loop
    pbar = tqdm(total=max_training_timesteps - time_step)
    while time_step <= max_training_timesteps:

        if args.env == 'getout':
            env = getout_utils.create_getout_instance(args)
            epsilon = max(math.exp(-i_episode / 0.02), 0.02)
        elif args.m == 'atari':
            obs, info = env.reset()
            epsilon = max(math.exp(-i_episode / 0.02), 0.02)
        elif args.m == "loot" or args.m == "threefish":
            epsilon = max(math.exp(-i_episode / 500), 0.02)
        else:
            raise ValueError
        #  initialize game
        # state = utils_getout.extract_logic_state_getout(env, args)
        # state[:, :, -2:] = state[:, :, -2:] / 50
        current_ep_reward = 0

        # epsilon = epsilon_func(i_episode)

        for t in range(1, max_ep_len + 1):
            # limit frame rate
            if args.with_explain:
                current_frame_time = time.time()
                if env_args.last_frame_time + env_args.target_frame_duration > current_frame_time:
                    sl = (env_args.last_frame_time + env_args.target_frame_duration) - current_frame_time
                    time.sleep(sl)
                    continue
                env_args.last_frame_time = current_frame_time  # save frame start time for next iteration

            # select action with policy
            if args.m == "loot" or args.m=="threefish":
                action = agent.select_action(obs, epsilon=epsilon)
            else:
                action = agent.select_action(env, epsilon=epsilon)
            if args.m == 'getout':
                obs = np.array(env.camera.screen.convert("RGB"))
                reward = env_step(action, env, args)
                done = env.level.terminated
            elif args.m == 'atari':
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            elif args.m == "loot":
                env.act(action)
                reward, obs, done = env.observe()
                reward = reward.tolist()[0]
            elif args.m == "threefish":
                env.act(action)
                reward, obs, done = env.observe()
                reward = reward.tolist()[0]
            else:
                raise ValueError
            # print(action)
            if args.m == "atari":
                state = env.objects
            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            # if reward:
            #     print("REWARD! :", reward)

            time_step += 1
            pbar.update(1)
            rtpt.step()
            current_ep_reward += reward
            # update PPO agent
            if time_step % update_timestep == 0:
                agent.update()
            # if time_step % 10 == 0:
            #     import matplotlib.pyplot as plt
            #     plt.imshow(env._get_obs())
            #     plt.show()
            if args.with_explain:
                screen_text = f"act: {args.action_names[action - 1]} re: {reward}"
                # Red
                env_args.obs = obs
                env_args.obs[:10, :10] = 0
                env_args.obs[:10, :10, 0] = 255
                # Blue
                env_args.obs[:10, 10:20] = 0
                env_args.obs[:10, 10:20, 2] = 255
                draw_utils.addCustomText(env_args.obs, f"{args.teacher_agent}",
                                         color=(255, 255, 255), thickness=1, font_size=0.2, pos=[2, 5])
                game_plot = draw_utils.rgb_to_bgr(env_args.obs)
                screen_plot = draw_utils.image_resize(game_plot,
                                                      int(game_plot.shape[0] * env_args.zoom_in),
                                                      int(game_plot.shape[1] * env_args.zoom_in))
                draw_utils.addText(screen_plot, screen_text,
                                   color=(255, 228, 181), thickness=2, font_size=0.6, pos="upper_right")
                video_out = draw_utils.write_video_frame(video_out, screen_plot)
            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))
                wandb.log({'reward': print_avg_reward}, step=time_step)
                print_running_reward = 0
                print_running_episodes = 0

                step_list.append([time_step])
                reward_list.append([print_avg_reward])
                if args.alg == 'logic':
                    weights_list.append([agent.get_weights()])

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = directory / f"{args.alg}_{args.env}_step_{time_step}.pth"
                print(f"saving model at : {str(checkpoint_path)}")
                if args.alg == 'logic':
                    agent.save(checkpoint_path, directory, step_list, reward_list, weights_list)
                else:
                    agent.save(checkpoint_path, directory, step_list, reward_list)
                print("model saved")
                print("Elapsed Time  : ", time.time() - start_time)
                print("--------------------------------------------------------------------------------------------")

                # save image of weights
                # if args.plot:
                #     if args.alg == 'logic':
                #         plot_weights(agent.get_weights(), image_directory, time_step)

            # break; if the episode is over
            if done:
                # print("Game over. New episode.")
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        i_episode += 1
        writer.add_scalar('Episode reward', current_ep_reward, i_episode)
        writer.add_scalar('Epsilon', epsilon, i_episode)

    # env.close()
    wandb.finish()
    # print total training time
    print("============================================================================================")
    with open(args.trained_model_folder / 'data.csv', 'w', newline='') as f:
        dataset = csv.writer(f)
        header = ('steps', 'reward')
        dataset.writerow(header)
        data = np.hstack((step_list, reward_list))
        for row in data:
            dataset.writerow(row)
    if args.alg == 'logic':
        with open(args.trained_model_folder / 'weights.csv', 'w', newline='') as f:
            dataset = csv.writer(f)
            for row in weights_list:
                dataset.writerow(row)

    end_time = time.time()
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    main()
