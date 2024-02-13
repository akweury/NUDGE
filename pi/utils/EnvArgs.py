# Created by jing at 09.02.24
import torch


class EnvArgs():
    """ generate one micro-program
    """

    def __init__(self, args, game_num, window_size, fps):
        super().__init__()
        self.output_folder = args.output_folder
        self.game_num = game_num
        self.game_i = 0
        self.win_count = 0
        self.zoom_in = args.zoom_in
        self.win_rate = torch.zeros(2, game_num)
        self.width_game_window = int(window_size[1] * args.zoom_in)
        self.height_game_window = int(window_size[0] * args.zoom_in)
        self.width_left_panel = int(window_size[0] * 0.5 * args.zoom_in)
        self.width_right_panel = int(window_size[1] * 0.25 * args.zoom_in)
        self.zoom_in = args.zoom_in
        # frame rate limiting
        self.fps = fps
        self.target_frame_duration = 1 / fps
        self.last_frame_time = 0
        self.db_num = 4
        self.current_lives = args.max_lives
        self.reward_lost_one_live = args.reward_lost_one_live
        self.current_steak = 0
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
