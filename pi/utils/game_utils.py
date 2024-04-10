# Created by jing at 18.01.24
import json
from functools import partial
from gzip import GzipFile
from pathlib import Path
import torch
import cv2 as cv
import numpy as np
from torch import nn
import ocatari

from pi.Player import SymbolicMicroProgramPlayer, PpoPlayer, OCA_PPOAgent
from pi.ale_env import ALEModern

from pi.utils import draw_utils, math_utils, file_utils
from src.agents.random_agent import RandomPlayer


class RolloutBuffer:
    def __init__(self, filename):
        self.filename = filename
        self.row_names = []
        self.win_rate = 0
        self.actions = []
        self.lost_actions = []
        self.game_next_states = []
        self.logic_states = []
        self.lost_logic_states = []
        self.neural_states = []
        self.action_probs = []
        self.logprobs = []
        self.rewards = []
        self.lost_rewards = []
        self.ungrounded_rewards = []
        self.terminated = []
        self.predictions = []
        self.reason_source = []
        self.game_number = []

    def clear(self):
        del self.win_rate
        del self.actions[:]
        del self.lost_actions[:]
        del self.game_next_states[:]
        del self.logic_states[:]
        del self.lost_logic_states[:]
        del self.neural_states[:]
        del self.action_probs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.lost_rewards[:]
        del self.ungrounded_rewards[:]
        del self.terminated[:]
        del self.predictions[:]
        del self.reason_source[:]
        del self.game_number[:]

    def load_buffer(self, args):
        with open(self.filename, 'r') as f:
            state_info = json.load(f)
        print(f"==> Loaded game buffer file: {self.filename}")

        self.win_rates = torch.tensor(state_info['win_rates'])
        self.actions = [torch.tensor(state_info['actions'][i]) for i in range(len(state_info['actions']))]
        self.logic_states = [torch.tensor(state_info['logic_states'][i]) for i in
                             range(len(state_info['logic_states']))]
        if state_info['next_states'][0][0] is not None:
            self.game_next_states = [torch.tensor(state_info['next_states'][i]) for i in
                                     range(len(state_info['next_states']))]
        self.rewards = [torch.tensor(state_info['reward'][i]) for i in range(len(state_info['reward']))]
        self.row_names = state_info['row_names']
        if 'neural_states' in list(state_info.keys()):
            self.neural_states = torch.tensor(state_info['neural_states']).to(args.device)
        if 'action_probs' in list(state_info.keys()):
            self.action_probs = torch.tensor(state_info['action_probs']).to(args.device)
        if 'logprobs' in list(state_info.keys()):
            self.logprobs = torch.tensor(state_info['logprobs']).to(args.device)
        if 'terminated' in list(state_info.keys()):
            self.terminated = torch.tensor(state_info['terminated']).to(args.device)
        if 'predictions' in list(state_info.keys()):
            self.predictions = torch.tensor(state_info['predictions']).to(args.device)
        if 'ungrounded_rewards' in list(state_info.keys()):
            self.ungrounded_rewards = state_info['ungrounded_rewards']
        if 'game_number' in list(state_info.keys()):
            self.game_number = state_info['game_number']
        if "reason_source" in list(state_info.keys()):
            self.reason_source = state_info['reason_source']
        else:
            self.reason_source = ["neural"] * len(self.actions)

    def save_data(self):
        data = {'actions': self.actions,
                'next_states': self.game_next_states,
                'logic_states': self.logic_states,
                'neural_states': self.neural_states,
                'action_probs': self.action_probs,
                'logprobs': self.logprobs,
                'reward': self.rewards,
                'ungrounded_rewards': self.ungrounded_rewards,
                'terminated': self.terminated,
                'predictions': self.predictions,
                "reason_source": self.reason_source,
                'game_number': self.game_number,
                'row_names': self.row_names,
                'win_rates': self.win_rates
                }

        with open(self.filename, 'w') as f:
            json.dump(data, f)
        print(f'data saved in file {self.filename}')


def load_buffer(args, buffer_filename):
    buffer = RolloutBuffer(buffer_filename)
    buffer.load_buffer(args)
    return buffer


def _load_checkpoint(fpath, device):
    fpath = Path(fpath)
    with fpath.open("rb") as file:
        with GzipFile(fileobj=file) as inflated:
            return torch.load(inflated, map_location=device)


def _epsilon_greedy(obs, model, eps=0.001):
    if torch.rand((1,)).item() < eps:
        return torch.randint(model.action_no, (1,)).item(), None
    q_val, argmax_a = model(obs).max(1)
    return argmax_a.item(), q_val


def zoom_image(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def get_game_viewer(env_args):
    width = int(env_args.width_game_window + env_args.width_left_panel)
    height = int(env_args.zoom_in * env_args.height_game_window)
    out = draw_utils.create_video_out(width, height)
    return out


def plot_game_frame(agent_type, env_args, out, obs, analysis_plot, screen_text):
    # Red
    obs[:10, :10] = 0
    obs[:10, :10, 0] = 255
    # Blue
    obs[:10, 10:20] = 0
    obs[:10, 10:20, 2] = 255
    draw_utils.addCustomText(obs, agent_type,
                             color=(255, 255, 255), thickness=1, font_size=0.3, pos=[1, 5])
    game_plot = draw_utils.rgb_to_bgr(obs)
    analysis_plot = draw_utils.rgb_to_bgr(analysis_plot)

    screen_plot = draw_utils.image_resize(game_plot,
                                          int(game_plot.shape[0] * env_args.zoom_in),
                                          int(game_plot.shape[1] * env_args.zoom_in))
    draw_utils.addText(screen_plot, screen_text,
                       color=(255, 228, 181), thickness=2, font_size=0.6, pos="upper_right")
    # label position
    for o_i in range(len(env_args.logic_state)):
        o_pos = env_args.logic_state[o_i][-2:]
        if torch.tensor(o_pos).sum() > 0:
            pixel_x = int(screen_plot.shape[1] * o_pos[0])
            pixel_y = int(screen_plot.shape[0] * o_pos[1])
            text = f"{o_pos[0]:.2f}, {o_pos[1]:.2f}"
            if o_pos[0] < 0 or o_pos[1] < 0:
                o_pos = [10, 10]
            # print(f"{text} {screen_plot.shape}")
            draw_utils.addText(screen_plot, text, color=(255, 255, 255), thickness=1, font_size=0.5,
                               pos=[pixel_x, pixel_y])

    # smp explanations
    pixel_x = int(screen_plot.shape[1] * 0.05)
    pixel_y = int(screen_plot.shape[0] * 0.6)
    draw_utils.addText(screen_plot, env_args.explain_text, color=(255, 255, 255), thickness=1, font_size=0.5,
                       pos=[pixel_x, pixel_y])

    # explain_plot_four_channel = draw_utils.three_to_four_channel(explain_plot)
    screen_with_explain = draw_utils.hconcat_resize([screen_plot, analysis_plot])
    out = draw_utils.write_video_frame(out, screen_with_explain)
    if env_args.save_frame:
        draw_utils.save_np_as_img(screen_with_explain,
                                  env_args.output_folder / "frames" / f"g_{env_args.game_i}_f_{env_args.frame_i}.png")

    return out, screen_with_explain


def update_game_args(frame_i, env_args):
    if env_args.reward < 0:
        env_args.score_update = True
    elif env_args.reward > 0:
        env_args.current_steak += 1
        env_args.max_steak = max(env_args.max_steak, env_args.current_steak)
        if env_args.max_steak >= 2 and not env_args.has_win_2:
            env_args.has_win_2 = True
            env_args.win_2 = env_args.game_i
        if env_args.max_steak >= 3 and not env_args.has_win_3:
            env_args.has_win_3 = True
            env_args.win_3 = env_args.game_i
        if env_args.max_steak >= 5 and not env_args.has_win_5:
            env_args.has_win_5 = True
            env_args.win_5 = env_args.game_i
        env_args.score_update = True
    else:
        env_args.score_update = False
    frame_i += 1

    return frame_i


def kangaroo_patches(env_args, reward, lives):
    env_args.score_update = False
    if lives < env_args.current_lives:
        reward += env_args.reward_lost_one_live
        env_args.current_lives = lives
        env_args.score_update = True

    return reward


def plot_mt_asterix(env_args, agent):
    if agent.agent_type == "smp":
        explain_str = ""
        for i, beh_i in enumerate(env_args.explaining['behavior_index']):
            try:
                explain_str += (f"{env_args.explaining['behavior_conf'][i]:.1f} "
                                f"{agent.behaviors[beh_i].beh_type} {agent.behaviors[beh_i].clause}\n")
                if i > 5:
                    break
            except IndexError:
                print("")
        data = (f"Max steaks: {env_args.max_steak}\n"
                f"(Frame) Behavior act: {agent.args.action_names[env_args.action]}\n"
                f"{explain_str}"
                f"# PF Behaviors: {len(agent.pf_behaviors)}\n"
                f"# Def Behaviors: {len(agent.def_behaviors)}\n"
                f"# Att Behaviors: {len(agent.att_behaviors)}\n"
                f"# Att Skill Behaviors: {len(agent.skill_att_behaviors)}\n")

    else:
        data = (f"Max steaks: {env_args.max_steak}\n"
                f"Win 2 steaks at ep: {env_args.win_2}\n"
                f"Win 3 steaks at ep: {env_args.win_3}\n"
                f"Win 5 steaks at ep: {env_args.win_5}\n")
    # plot game frame
    mt_plot = draw_utils.visual_info(data, height=env_args.height_game_window, width=env_args.width_left_panel,
                                     font_size=0.5,
                                     text_pos=[20, 20])
    return mt_plot


def plot_wr(env_args):
    if env_args.score_update or env_args.wr_plot is None:
        wr_plot = draw_utils.plot_line_chart(env_args.win_rate[:env_args.game_i].unsqueeze(0),
                                             env_args.output_folder, ['smp', 'ppo'],
                                             title='win_rate', cla_leg=True, figure_size=(30, 10))
        env_args.wr_plot = wr_plot
    else:
        wr_plot = env_args.wr_plot
    return wr_plot


class AtariNet(nn.Module):
    """ Estimator used by DQN-style algorithms for ATARI games.
        Works with DQN, M-DQN and C51.
    """

    def __init__(self, action_no, distributional=False):
        super().__init__()

        self.action_no = out_size = action_no
        self.distributional = distributional

        # configure the support if distributional
        if distributional:
            support = torch.linspace(-10, 10, 51)
            self.__support = nn.Parameter(support, requires_grad=False)
            out_size = action_no * len(self.__support)

        # get the feature extractor and fully connected layers
        self.__features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.__head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Linear(512, out_size),
        )

    def forward(self, x):
        assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div(255)

        x = self.__features(x)
        qs = self.__head(x.view(x.size(0), -1))

        if self.distributional:
            logits = qs.view(qs.shape[0], self.action_no, len(self.__support))
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.__support.expand_as(qs_probs)).sum(2)
        return qs


def create_agent(args, agent_type):
    #### create agent
    if agent_type == "smp":
        agent = SymbolicMicroProgramPlayer(args)
    elif agent_type == 'random':
        agent = RandomPlayer(args)
    elif agent_type == 'human':
        agent = 'human'
    elif agent_type == "ppo":
        agent = PpoPlayer(args)
    elif agent_type == "oca_ppo":

        agent = OCA_PPOAgent(args.num_actions)
        ckpt = torch.load(args.model_path, args.device)["model_weights"]
        agent.load_state_dict(ckpt)

    elif agent_type in ['pretrained', 'DQN-A', 'DQN-T', 'DQN-R']:
        # game/seed/model
        game_name = args.m.lower()
        ckpt = _load_checkpoint(args.model_path, args.device)
        # set env
        env = ALEModern(
            game_name,
            torch.randint(100_000, (1,)).item(),
            sdl=False,
            device=args.device,
            clip_rewards_val=False,
            record_dir=None,
        )

        # init model
        model = AtariNet(env.action_space.n, distributional="C51_" in str(args.model_path))
        model.load_state_dict(ckpt["estimator_state"])
        model = model.to(args.device)
        # configure policy
        policy = partial(_epsilon_greedy, model=model, eps=0.001)
        agent = policy
    else:
        raise ValueError
    agent.agent_type = agent_type
    return agent


def screen_shot(env_args, video_out, obs, wr_plot, mt_plot, db_plots, dead_counter, screen_text):
    _, screen_with_explain = plot_game_frame(env_args, video_out, obs, wr_plot, screen_text)
    file_name = str(env_args.output_folder / f"screen_{dead_counter}.png")
    draw_utils.save_np_as_img(screen_with_explain, file_name)


def game_over_log(args, agent, env_args):
    if args.m == "Asterix":
        game_score = env_args.game_rewards[-1]
        env_args.win_rate[env_args.game_i] = sum(game_score)
        print(
            f"- Ep: {env_args.game_i}, Win: {env_args.win_rate[env_args.game_i]} "
            f"Ep Score: {sum(game_score)} Ep Loss: {env_args.state_loss}")
    elif args.m == "Pong":
        game_score = env_args.game_rewards[-1]
        env_args.win_rate[env_args.game_i] = sum(game_score)
        print(
            f"- Ep: {env_args.game_i}, Win: {sum(env_args.win_rate[:env_args.game_i] > 0)}/{env_args.game_i} "
            f"Ep Score: {sum(game_score)} Ep Loss: {env_args.state_loss}")
    elif args.m == "Kangaroo":
        game_score = env_args.game_rewards[-1]
        env_args.win_rate[env_args.game_i] = sum(game_score)
        print(
            f"- Ep: {env_args.game_i}, Win:  {env_args.win_rate[env_args.game_i]} "
            f"Ep Score: {sum(game_score)} Ep Loss: {env_args.state_loss}")
    elif args.m in ["Freeway", "Boxing"]:
        game_score = env_args.game_rewards[-1]
        env_args.win_rate[env_args.game_i] = sum(game_score)
        print(
            f"- Ep: {env_args.game_i}, Win:  {env_args.win_rate[env_args.game_i]} "
            f"Ep Score: {sum(game_score)} Ep Loss: {env_args.state_loss}")
    else:
        raise ValueError

    # if agent.agent_type == "pretrained" or agent.agent_type == "ppo":
    #     draw_utils.plot_line_chart(env_args.win_rate.unsqueeze(0)[:, :env_args.game_i], args.check_point_path,
    #                                [agent.agent_type], title=f"wr_{agent.agent_type}_{len(env_args.win_rate)}")
    # if agent.agent_type == "smp":
    #
    #     pretrained_wr = torch.load(args.output_folder / f"wr_pretrained_{args.teacher_game_nums}.pt")
    #     smp_wr = env_args.win_rate.unsqueeze(0)[:, :env_args.game_i]
    #     if len(pretrained_wr) < smp_wr.shape[1]:
    #         pretrained_wr = torch.cat((pretrained_wr, torch.zeros(10000)))
    #     all_wr = torch.cat((pretrained_wr.unsqueeze(0)[:, :smp_wr.shape[1]], smp_wr), dim=0)
    #     draw_utils.plot_line_chart(all_wr, args.check_point_path,
    #                                ["pretrained", agent.agent_type],
    #                                title=f"wr_{agent.agent_type}_{len(env_args.win_rate)}")
    #     for b_i, beh in enumerate(agent.def_behaviors):
    #         print(f"- DefBeh {b_i}/{len(agent.def_behaviors)}: {beh.clause}")
    #     for b_i, beh in enumerate(agent.att_behaviors):
    #         print(f"+ AttBeh {b_i}/{len(agent.att_behaviors)} : {beh.clause}")
    #     for b_i, beh in enumerate(agent.pf_behaviors):
    #         print(f"~ PfBeh {b_i}/{len(agent.pf_behaviors)}: {beh.clause}")


def frame_log(agent, env_args):
    # game log
    if agent.agent_type == "smp":
        if env_args.explaining is not None:
            for beh_i in env_args.explaining['behavior_index']:
                print(
                    f"({agent.agent_type})g: {env_args.game_i} f: {env_args.frame_i}, rw: {env_args.reward}, act: {env_args.action}, "
                    f"behavior: {agent.behaviors[beh_i].clause}")


def revise_loss_log(env_args, agent, video_out):
    screen_text = f"Ep: {env_args.game_i}, Best: {env_args.best_score}"
    mt_plot = plot_mt_asterix(env_args, agent)
    screen_shot(env_args, video_out, env_args.obs, None, mt_plot, [], env_args.dead_counter, screen_text)


def save_game_buffer(args, env_args, buffer_filename):
    buffer = RolloutBuffer(buffer_filename)
    buffer.game_next_states = env_args.game_next_states
    buffer.logic_states = env_args.game_states
    buffer.actions = env_args.game_actions
    buffer.rewards = env_args.game_rewards
    buffer.win_rates = env_args.win_rate.tolist()
    buffer.row_names = args.row_names
    buffer.save_data()


def finish_one_run(env_args, args, agent):
    draw_utils.plot_line_chart(env_args.win_rate.unsqueeze(0), args.check_point_path,
                               [agent.agent_type], title=f"wr_{agent.agent_type}_{len(env_args.win_rate)}")
    torch.save(env_args.win_rate, args.check_point_path / f"wr_{agent.agent_type}_{len(env_args.win_rate)}.pt")


def get_ocname(m):
    if m == "montezuma_revenge":
        return "MontezumaRevenge"
    elif m == "Kangaroo":
        return "Kangaroo"
    elif m == "fishing_derby":
        return "FishingDerby"
    elif m == "Pong":
        return "Pong"
    elif m == "Asterix":
        return "Asterix"
    elif m == "Freeway":
        return "Freeway"
    elif m == "Boxing":
        return "Boxing"
    else:
        raise ValueError


from ocatari.core import OCAtari
from pi.utils.EnvArgs import EnvArgs
from tqdm import tqdm
import time

from pi.utils.atari import game_patches
from pi.utils.oc_utils import extract_logic_state_atari


def collect_data_dqn_a(agent, args, buffer_filename, save_buffer):
    oc_name = get_ocname(args.m)
    env = OCAtari(oc_name, mode="revised", hud=True, render_mode='rgb_array')
    obs, info = env.reset()
    env_args = EnvArgs(agent=agent, args=args, window_size=obs.shape[:2], fps=60)
    agent.position_norm_factor = obs.shape[0]
    if args.with_explain:
        video_out = get_game_viewer(env_args)

    for game_i in tqdm(range(args.teacher_game_nums), desc=f"Collecting GameBuffer by {agent.agent_type}"):
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
                if agent.agent_type == "oca_ppo":
                    env_args.action = agent.draw_action(env.dqn_obs.to(env_args.device)).item()
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
            if sum(env_args.rewards) > 0:
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

        game_over_log(args, agent, env_args)
        env_args.reset_buffer_game()
    env.close()
    finish_one_run(env_args, args, agent)
    if args.with_explain:
        draw_utils.release_video(video_out)
    if save_buffer:
        save_game_buffer(args, env_args, buffer_filename)


def load_atari_buffer(args, buffer_filename):
    buffer = load_buffer(args, buffer_filename)
    print(f'- Loaded game history : {len(buffer.logic_states)}')
    buffer_win_rates = buffer.win_rates
    row_names = buffer.row_names
    game_num = len(buffer.actions)
    actions = []
    rewards = []
    states = []
    next_states = []

    for g_i in range(game_num):
        actions.append(buffer.actions[g_i].to(args.device))
        rewards.append(buffer.rewards[g_i].to(args.device))
        states.append(buffer.logic_states[g_i].to(args.device))
        # self.next_states.append(buffer.game_next_states[g_i].to(self.args.device))

    return {"states": states, "actions": actions, "rewards": rewards}
