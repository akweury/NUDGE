# Created by jing at 06.02.24

"""A simple class for viewing images using pyglet."""
import torch

from pi.utils import math_utils


# Original code from the nes_py project

class ImageViewer(object):
    """A simple class for viewing images using pyglet."""

    def __init__(self, caption, height, width,
                 monitor_keyboard=False,
                 relevant_keys=None
                 ):
        """
        Initialize a new image viewer.

        Args:
            caption (str): the caption/title for the window
            height (int): the height of the window
            width (int): the width of the window
            monitor_keyboard: whether to monitor events from the keyboard
            relevant_keys: the relevant keys to monitor events from

        Returns:
            None
        """
        # detect if rendering from python threads and fail
        import threading
        if threading.current_thread() is not threading.main_thread():
            msg = 'rendering from python threads is not supported'
            raise RuntimeError(msg)
        # import pyglet within class scope to resolve issues with how pyglet
        # interacts with OpenGL while using multiprocessing
        import pyglet
        self.pyglet = pyglet
        # a mapping from pyglet key identifiers to native identifiers
        self.KEY_MAP = {
            self.pyglet.window.key.ENTER: ord('\r'),
            self.pyglet.window.key.SPACE: ord(' '),
        }
        self.caption = caption
        self.height = height
        self.width = width
        self.monitor_keyboard = monitor_keyboard
        self.relevant_keys = relevant_keys
        self._window = None
        self._pressed_keys = []
        self._is_escape_pressed = False

    @property
    def is_open(self):
        """Return a boolean determining if this window is open."""
        return self._window is not None

    @property
    def is_escape_pressed(self):
        """Return True if the escape key is pressed."""
        return self._is_escape_pressed

    @property
    def pressed_keys(self):
        """Return a sorted list of the pressed keys."""
        return tuple(sorted(self._pressed_keys))

    def _handle_key_event(self, symbol, is_press):
        """
        Handle a key event.

        Args:
            symbol: the symbol in the event
            is_press: whether the event is a press or release

        Returns:
            None

        """
        # remap the key to the expected domain
        symbol = self.KEY_MAP.get(symbol, symbol)
        # check if the symbol is the escape key
        if symbol == self.pyglet.window.key.ESCAPE:
            self._is_escape_pressed = is_press
            return
        # make sure the symbol is relevant
        if self.relevant_keys is not None and symbol not in self.relevant_keys:
            return
        # handle the press / release by appending / removing the key to pressed
        if is_press:
            self._pressed_keys.append(symbol)
        else:
            try:
                self._pressed_keys.remove(symbol)
            except ValueError:
                pass

    def on_key_press(self, symbol, modifiers):
        """Respond to a key press on the keyboard."""
        self._handle_key_event(symbol, True)

    def on_key_release(self, symbol, modifiers):
        """Respond to a key release on the keyboard."""
        self._handle_key_event(symbol, False)

    def open(self):
        """Open the window."""
        # create a window for this image viewer instance
        self._window = self.pyglet.window.Window(
            caption=self.caption,
            height=self.height,
            width=self.width,
            vsync=False,
            resizable=True,
        )

        # add keyboard event monitors if enabled
        if self.monitor_keyboard:
            self._window.event(self.on_key_press)
            self._window.event(self.on_key_release)

    def close(self):
        """Close the window."""
        if self.is_open:
            self._window.close()
            self._window = None

    def show(self, frame):
        """
        Show an array of pixels on the window.

        Args:
            frame (numpy.ndarray): the frame to show on the window

        Returns:
            None
        """
        # check that the frame has the correct dimensions
        if len(frame.shape) != 3:
            raise ValueError('frame should have shape with only 3 dimensions')
        # open the window if it isn't open already
        if not self.is_open:
            self.open()
        # prepare the window for the next frame
        self._window.clear()
        self._window.switch_to()
        self._window.dispatch_events()
        # create an image data object
        image = self.pyglet.image.ImageData(
            frame.shape[1],
            frame.shape[0],
            'RGB',
            frame.tobytes(),
            pitch=frame.shape[1] * -3
        )
        # send the image to the window
        image.blit(0, 0, width=self._window.width, height=self._window.height)
        self._window.flip()


# explicitly define the outward facing API of this module
__all__ = [ImageViewer.__name__]


def patch_asterix(game_info, states, actions, rewards, lives):
    new_states = remove_last_key_frame(game_info, states)
    new_actions = actions[:len(new_states)]
    new_rewards = rewards[:len(new_states)]
    new_rewards[-1] = rewards[-1]
    game_over = False
    if lives == 0:
        game_over = True
    agent_pos = torch.tensor(new_states[-1][0][-2:]).unsqueeze(0)
    enemy_pos = torch.tensor(new_states[-1][1:9])[:, -2:]
    failed_dist = torch.abs(enemy_pos - agent_pos).sum(dim=1).min()

    if failed_dist > 0.2:
        new_states = new_states[:-10]
        new_actions = actions[:len(new_states)]
        new_rewards = rewards[:len(new_states)]
    else:
        print(f"agent-enemy failed dist: {failed_dist}")

    return new_states, new_actions, new_rewards, game_over


def frame_back_stop_drop(states):
    states = torch.tensor(states)
    key_frame_index = None
    for s_i in reversed(range(len(states))):
        print(s_i)
        delta = (states[s_i] - states[s_i - 1])[-6:]
        exist_missing_projectile = delta[:, [-3, -4]].sum(dim=-1) == -1
        if exist_missing_projectile.sum() > 0:
            not_left_border = states[s_i - 1, -6:][exist_missing_projectile][:, -2] > 0.1
            not_right_border = states[s_i - 1, -6:][exist_missing_projectile][:, -2] < 0.7
            not_bottom_border = states[s_i - 1, -6:][exist_missing_projectile][:, -1] < 0.8
            if not_left_border.prod() and not_right_border.prod() and not_bottom_border.prod():
                key_frame_index = s_i
                break
    return key_frame_index


def patch_kangaroo(states):
    key_frame_index = frame_back_stop_drop(states)
    return states[:key_frame_index]


def frame_back_fire_action(states, actions, rewards, action_names):
    for a_i in reversed(range(len(actions))):
        if "fire" in action_names[actions[a_i]]:
            rewards[a_i] = rewards[-1]
            rewards = rewards[:a_i + 1]
            states = states[:a_i + 1]
            actions = actions[:a_i + 1]
            break
    return states, actions, rewards


def frame_patch_boxing(states, actions, rewards, action_names):
    if len(rewards) > 0 and rewards[-1] > 0:
        states, actions, rewards = frame_back_fire_action(states, actions, rewards, action_names)

    return states, actions, rewards


def atari_frame_patches(args, env_args, info):
    if args.m == 'Boxing':
        env_args.logic_states, env_args.actions, env_args.rewards = frame_patch_boxing(env_args.logic_states,
                                                                                       env_args.actions,
                                                                                       env_args.rewards,
                                                                                       args.action_names)
        reward_tensor = torch.tensor(env_args.rewards)
        reward_tensor[reward_tensor > 0].sum()
        env_args.state_score = reward_tensor[reward_tensor > 0].sum()
        if env_args.terminated or env_args.truncated:
            env_args.game_over = True


def atari_patches(args, env_args, info):
    if args.m == "Asterix":
        env_args.logic_states, env_args.actions, env_args.rewards, env_args.game_over = patch_asterix(
            args.game_info, env_args.logic_states, env_args.actions, env_args.rewards, info['lives'])
    if args.m == 'Boxing':
        reward_tensor = torch.tensor(env_args.rewards)
        reward_tensor[reward_tensor > 0].sum()
        env_args.state_score = reward_tensor[reward_tensor > 0].sum()
        if env_args.terminated or env_args.truncated:
            env_args.game_over = True
    if args.m == 'Kangaroo':
        new_states = patch_kangaroo(env_args.logic_states)
        env_args.states = new_states
        if env_args.terminated or env_args.truncated:
            env_args.game_over = True
    if args.m == "Breakout":
        if env_args.terminated or env_args.truncated:
            env_args.game_over = True
            env_args.new_life = True
            reward_tensor = torch.tensor(env_args.rewards)
            reward_tensor[reward_tensor > 0].sum()
            env_args.state_score = reward_tensor[reward_tensor > 0].sum()
    if args.m == "Freeway":
        if env_args.terminated or env_args.truncated:
            env_args.game_over = True
            env_args.new_life = True
            reward_tensor = torch.tensor(env_args.rewards)
            reward_tensor[reward_tensor > 0].sum()
            env_args.state_score = reward_tensor[reward_tensor > 0].sum()


def patch_boxing(actions, rewards, action_names):
    new_rewards = shift_reward_to_attack_actions(actions, rewards, action_names)
    return new_rewards


def remove_last_key_frame(game_info, states, max_dist=35):
    last_frame_enemy_num = torch.tensor(states[-1])[:, 1].sum()
    not_found = True
    state_i = len(states) - 1
    while not_found:
        frame_enemy_num = torch.tensor(states[state_i])[:, 1].sum()
        if frame_enemy_num != last_frame_enemy_num:
            break
        else:
            state_i -= 1
    new_states = states[:state_i + 1]

    pos_indices = [game_info["axis_x_col"], game_info["axis_y_col"]]
    test_state = torch.tensor(new_states[-1])
    pos_agent = test_state[0, pos_indices]
    enemy_indices = test_state[:, 1] > 0
    pos_enemy = test_state[enemy_indices][:, pos_indices]
    # dist, _ = math_utils.dist_a_and_b_closest(pos_agent, pos_enemy, dir)

    # if dist.sum(dim=-1).min() > max_dist:
    #     print(f"dist:{dist.sum(dim=-1).min()}")
    # raise ValueError
    return new_states


def shift_reward_to_attack_actions(actions, rewards, action_names):
    for r_i, reward in enumerate(rewards):
        if reward > 0:
            for reverse_r_i in reversed(range(r_i)):
                if "fire" in action_names[actions[reverse_r_i]]:
                    rewards[reverse_r_i] = rewards[r_i]
                    rewards[r_i] = 0
                    break
    return rewards
