# Created by jing at 09.02.24
import torch


class EnvArgs():
    """ generate one micro-program
    """

    def __init__(self, args, window_size, fps):
        super().__init__()
        # game setting
        self.output_folder = args.output_folder
        self.max_lives = args.max_lives
        self.reward_lost_one_live = args.reward_lost_one_live
        # layout setting
        self.zoom_in = args.zoom_in
        self.db_num = 4
        self.width_game_window = int(window_size[1] * args.zoom_in)
        self.height_game_window = int(window_size[0] * args.zoom_in)
        self.width_left_panel = int(window_size[0] * 0.5 * args.zoom_in)
        self.width_right_panel = int(window_size[1] * 0.25 * args.zoom_in)
        # frame rate limiting
        self.fps = fps
        self.target_frame_duration = 1 / fps
        self.last_frame_time = 0
        # record and statistical properties
        self.action = None
        self.logic_state = None
        self.reward = None
        self.obs = None
        self.game_states = []
        self.game_actions = []
        self.game_rewards = []
        self.game_num = args.game_nums
        self.game_i = 0
        self.win_count = 0
        self.dead_counter = 0
        self.current_steak = 0
        self.win_rate = torch.zeros(2, args.game_nums)
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

    def reset_args(self, game_i):
        self.game_i = game_i
        self.frame_i = 0
        self.current_lives = self.max_lives
        self.state_score = 0

    def update_args(self, env_args):
        if env_args.state_score > env_args.best_score:
            env_args.best_score = env_args.state_score

        if env_args.reward < 0:
            self.score_update = True
            self.current_steak = 0
        elif env_args.reward > 0:
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
        self.win_rate[0, self.game_i] = self.best_score
        self.rewards[-1] += self.reward_lost_one_live
        self.dead_counter += 1

    def buffer_frame(self):
        self.logic_states.append(self.logic_state)
        self.actions.append(self.action)
        self.rewards.append(self.reward)

    def buffer_game(self):
        self.game_states.append(self.logic_states)
        self.game_actions.append(self.actions)
        self.game_rewards.append(self.rewards)

    def reset_buffer_game(self):
        self.logic_states = []
        self.actions = []
        self.rewards = []

    def update_wr(self,agent_type, game_i):
        if agent_type == "smp":
            self.win_rate[1, game_i] = self.state_score
        else:
            self.win_rate[1, game_i] = self.state_score
