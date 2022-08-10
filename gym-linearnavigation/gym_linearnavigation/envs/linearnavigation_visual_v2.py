import gym
import numpy as np
import cv2
import logging
import random
import pickle
import math
from pathlib import Path
from gym import error, spaces, utils
from gym.utils import seeding
from copy import deepcopy
from typing import List


class LinearNavigationVisualV2(gym.Env):
    """Adaptation of the VR linear navigation task by Tennant 2018

    The agent needs to stop in the reward zone to receive the reward

    This version implements multi-track episodes.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, is_probe, num_middle_repetitions, n_trials=1, max_blackbox_padding=0, p_image_noise=0.01, max_reward_displacement=0, rz_start=460):
        self.action_space = spaces.Discrete(2)  # Step or stop
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        # args
        self.is_probe = is_probe
        self.p_image_noise = p_image_noise

        # Customisable parameters
        self.reward_zone_start = rz_start # Note that RZ start & end are normalised to first blackbox end
        self.reward_zone_end = 510
        self.reward_duration = 10  # (frames) - setting this to zero is essentially a probe trial
        self.track_repetitions = num_middle_repetitions
        self.max_blackbox_padding = max_blackbox_padding
        self.middle_roll_max = 0  # Max number to randomly roll the middle section by
        self.max_reward_displacement = max_reward_displacement

        # Step probability
        self.all_step_sizes = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]]
        self.step_probs = [0.025, 0.075, 0.80, 0.075, 0.025]

        self.n_trials = n_trials

        self.viewer = None

        frames_dir = Path(__file__).parents[0] / 'continuous_frames_by_section'
        self._frames_cache = {
            'start': self._initial_load_frames(frames_dir / 'start'),
            'middle': self._initial_load_frames(frames_dir / 'middle'),
            'end': self._initial_load_frames(frames_dir / 'end'),
        }

        print('Loaded frame cache:')
        [print(f'{k}: {len(v)} frames') for k, v in self._frames_cache.items()]

        # Initialise state tracking variables the lazy way
        self.reset()

    @staticmethod
    def _initial_load_frames(path):
        """Loads frames from disk

        Magic numbers here correspond to optimal frame boundaries
        """
        every_nth_frame = 50
        if 'middle' in str(path):
            frames = sorted([int(p.name.strip('.png')) for p in path.glob('*.png') if int(p.name.strip('.png')) < 4000])[::every_nth_frame]
        elif 'end' in str(path):
            frames = sorted(
                [int(p.name.strip('.png')) for p in path.glob('*.png') if int(p.name.strip('.png')) > 1000])[::every_nth_frame]
        elif 'start' in str(path):
            frames = sorted([int(p.name.strip('.png')) for p in path.glob('*.png') if int(p.name.strip('.png')) > 1500])[::every_nth_frame]
        else:
            raise ValueError('Frame paths are not correct')

        frames = [cv2.imread(str(path / (str(p) + '.png'))) for p in frames]

        return frames

    def _load_frames(self, section):
        """Returns cached frames"""
        return [f.copy() for f in self._frames_cache[section]]

    def _pad_blackbox(self, mode: str, frames: []) -> []:
        """Adds extra frames by a random amount to the end or start of the given frame list. Doubles the number added to the end"""
        if self.max_blackbox_padding == 0:
            return frames

        n_padding = np.random.randint(self.max_blackbox_padding)

        logging.debug(f'Padding frames ({mode}) of length {len(frames)} by {n_padding}')

        if mode == 'append':
            return frames + ([frames[-1]] * (2 * n_padding))
        elif mode == 'prepend':
            return ([frames[0]] * n_padding) + frames
        else:
            raise ValueError(f'Invalid blackbox padding mode {mode}')

    def _roll_frames(self, frames) -> []:
        """Rolls (rotates) the frames

        Wonky implementation means this mutates the underlying list
        """
        if self.middle_roll_max == 0:
            logging.debug(f'Frame rolling was turned off. Not rolling.')
            return frames

        n_times = np.random.randint(self.middle_roll_max)
        for i in range(n_times):
            frames.append(frames.pop(0))

        logging.debug(f'Rolled the middle section by {n_times} times')
        return frames

    def _load_track(self, track_repetitions: int) -> []:
        """Load the visual track representation"""

        # This won't work with egg distributions or anything fancy,
        # but it'll do for now...
        # See https://stackoverflow.com/questions/779495/access-data-in-package-subdirectory

        frames = []
        frames += self._pad_blackbox('prepend', self._load_frames('start'))
        logging.debug(f'Length of start frames: {len(frames)}')
        self.first_blackbox_end = len(frames)

        middle = []
        for _ in range(track_repetitions):
            middle += self._load_frames('middle')
        logging.debug(f'Length of repeated middle: {len(middle)}')
        middle = self._roll_frames(middle)

        frames.extend(middle)
        logging.debug(f'Length of start + repeated middle frames: {len(frames)}')

        frames += self._pad_blackbox('append', self._load_frames('end'))
        logging.debug(f'Total length track (frames): {len(frames)}')

        return frames

    def _get_image(self, readonly=False):
        if self.show_reward > 0 and not self.is_probe:
            logging.debug(f'Flashing the reward. Remaining duration : {self.show_reward}/{self.reward_duration}')

            if not readonly:
                self.show_reward -= 1
            else:
                logging.debug('Getting image readonly')

            current_frame = deepcopy(self.frames[self.location])
            current_frame[0:5] = 100  # Draw a grey bar at the top of the screen

            frame = current_frame
        else:
            try:
                frame = self.frames[self.location]
            except IndexError:
                logging.debug(f'Warning: Exceeded track bounds with position {self.location} and track length: {self.track_length}')
                frame = self.frames[-1]

        frame = deepcopy(frame)
        mask = np.random.choice(a=[False, True], size=self.observation_space.shape[0:2], p=[1-self.p_image_noise, self.p_image_noise])
        noise = np.random.permutation(self.frames[555])  # Make the noise similar to the actual colours
        frame[mask] = noise[mask]

        return frame

    def step(self, action):
        reward = 0

        in_reward_zone = (self.reward_zone_start + self.first_blackbox_end + self.reward_displacement) < self.location < (self.reward_zone_end + self.first_blackbox_end + self.reward_displacement)
        stopped = action == 0

        if stopped:
            if in_reward_zone and not self.reward_acquired:
                self.reward_acquired = True
                reward = 100

                self.show_reward = self.reward_duration
            else:
                reward = -1
        else:
            step_size = np.random.choice(self.step_sizes, p=self.step_probs)
            logging.debug(f'Stepping step sizes = {self.step_sizes}: {step_size}. {self.location} -> {self.location + step_size}')
            self.location += step_size

        self._episode_reward += reward

        trial_done = self.location >= self.track_length
        if trial_done:
            if self.trial != self.n_trials:
                done = False
                self._reset_trial()
            else:
                logging.info(f'All trials done. Reward {self._episode_reward}')
                done = True
        else:
            done = False

        logging.debug(f'{self.trial}: BB end: {self.first_blackbox_end}, Location: {self.location}, in rz: {in_reward_zone} reward acq: {self.reward_acquired}, frame reward: {reward}')

        return self._get_image(), reward, done, {}

    def _reset_trial(self) -> None:
        self.location = 0
        self.reward_acquired = False
        self.show_reward = 0

        self.step_sizes = random.choice(self.all_step_sizes)

        self.frames = self._load_track(self.track_repetitions)
        self.track_length = len(self.frames) - 1

        self.reward_displacement = random.randint(-self.max_reward_displacement, self.max_reward_displacement)

        self.trial += 1
        logging.debug(f'Resetting for trial {self.trial}')

    def reset(self):
        self.location = 0
        self.reward_acquired = False
        self.show_reward = 0

        self.step_sizes = random.choice(self.all_step_sizes)

        self.frames = self._load_track(self.track_repetitions)
        self.track_length = len(self.frames) - 1

        self.reward_displacement = random.randint(-self.max_reward_displacement, self.max_reward_displacement)

        self.trial = 1
        self._episode_reward = 0

        return self._get_image()

    def render(self, mode='human'):
        """RGB screen rendering function adapted (stolen) from Atari gyms"""
        img = self._get_image(readonly=True)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            img = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            cv2.imshow('n', img)
            cv2.waitKey(1)

    def close(self):
        pass