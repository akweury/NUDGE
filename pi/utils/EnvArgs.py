# Created by jing at 09.02.24
import torch
import shutil


class EnvArgs():
    """ generate one micro-program
    """

    def __init__(self, agent, args, window_size, fps):
        super().__init__()
        # game setting
        self.device = args.device
        self.save_frame = args.save_frame
        self.output_folder = args.game_buffer_path
        self.max_lives = args.max_lives
        self.reward_lost_one_live = args.reward_lost_one_live
        # layout setting
        self.zoom_in = args.zoom_in
        self.db_num = 4
        self.width_game_window = int(window_size[1] * args.zoom_in)
        self.height_game_window = int(window_size[0] * args.zoom_in)
        self.width_left_panel = int(window_size[0] * 2 * args.zoom_in)
        self.width_right_panel = int(window_size[1] * 0.25 * args.zoom_in)
        self.position_norm_factor = window_size[0]
        # frame rate limiting
        self.fps = fps
        self.target_frame_duration = 1 / fps
        self.last_frame_time = 0
        # record and statistical properties
        self.last_obs = torch.zeros((window_size[0], window_size[1], 3), dtype=torch.uint8).numpy()
        self.action = None
        self.logic_state = None
        self.reward = None
        self.obs = None
        self.game_states = []
        self.game_actions = []
        self.game_rewards = []
        self.game_i = 0
        self.win_count = 0
        self.dead_counter = 0
        self.current_steak = 0
        if agent.agent_type == "smp":
            self.game_num = args.student_game_nums
        elif agent.agent_type == "pretrained" or agent.agent_type == "ppo":
            self.game_num = args.teacher_game_nums
        else:
            raise ValueError
        self.win_rate = torch.zeros(self.game_num)
        self.win_2 = ""
        self.has_win_2 = False
        self.win_3 = ""
        self.has_win_3 = False
        self.win_5 = ""
        self.has_win_5 = False
        self.max_steak = 0
        self.wr_plot = None
        self.score_update = False
        self.best_score = 0
        self.mile_stone_scores = args.mile_stone_scores
        self.new_life = False

    def reset_args(self, game_i):
        self.game_i = game_i
        self.frame_i = 0
        self.current_lives = self.max_lives
        self.state_score = 0
        self.state_loss = 0
        self.game_over = False
        self.terminated = False
        self.truncated = False

    def update_args(self):
        self.frame_i += 1
        if self.state_score > self.best_score:
            self.best_score = self.state_score

        if self.reward < 0:
            self.score_update = True
            self.current_steak = 0
        elif self.reward > 0:
            self.current_steak += 1
            self.max_steak = max(self.max_steak, self.current_steak)
            if self.max_steak >= self.mile_stone_scores[0] and not self.has_win_2:
                self.has_win_2 = True
                self.win_2 = self.game_i
            if self.max_steak >= self.mile_stone_scores[1] and not self.has_win_3:
                self.has_win_3 = True
                self.win_3 = self.game_i
            if self.max_steak >= self.mile_stone_scores[2] and not self.has_win_5:
                self.has_win_5 = True
                self.win_5 = self.game_i
            self.score_update = True
        else:
            self.score_update = False

    def update_lost_live(self, current_live):
        self.current_lives = current_live
        self.score_update = True
        self.rewards[-1] += self.reward_lost_one_live
        self.dead_counter += 1

    def buffer_frame(self):
        self.logic_states.append(self.logic_state)
        self.actions.append(self.action)
        self.rewards.append(self.reward)

    def buffer_game(self, zero_reward, save_frame):
        states = []
        actions = []
        rewards = []
        for f_i, reward in enumerate(self.rewards):
            if f_i % 10 == 0 or reward != zero_reward:
                states.append(self.logic_states[f_i])
                actions.append(self.actions[f_i])
                rewards.append(self.rewards[f_i])
            if save_frame:
                # move dead frame to some folder
                shutil.copy2(self.output_folder /"frames" /f"g_{self.game_i}_f_{f_i}.png",
                             self.output_folder / "key_frames" / f"g_{self.game_i}_f_{f_i}.png")
        self.game_states.append(states)
        self.game_rewards.append(rewards)
        self.game_actions.append(actions)

    def reset_buffer_game(self):
        self.logic_states = []
        self.actions = []
        self.rewards = []
