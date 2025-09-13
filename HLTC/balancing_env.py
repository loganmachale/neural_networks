import math
import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv

class BalancingCartPoleEnv(gym.Env):
    """
    A modified Cart-Pole environment focused solely on balancing.
    The reward is based on the pole's angle, and the failure threshold is wider.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        super(BalancingCartPoleEnv, self).__init__()
        # Use the standard CartPole environment as the base
        self._env = gym.make('CartPole-v1', render_mode=render_mode)

        # --- Increase the pole angle threshold ---
        # Original was 12 degrees. New threshold is 12 + 15 = 27 degrees.
        new_threshold_radians = 27 * 2 * math.pi / 360
        self.unwrapped.theta_threshold_radians = new_threshold_radians
        # ----------------------------------------

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    @property
    def unwrapped(self):
        # Provides access to the underlying, unwrapped environment properties
        return self._env.unwrapped

    def step(self, action):
        obs, _, terminated, truncated, info = self._env.step(action)
        
        # New reward function focused on balancing
        pole_angle = obs[2]
        
        # Reward is higher the closer the pole is to vertical (angle = 0).
        # We use a cosine function: reward is 1 at 0 angle and decreases as it tilts.
        reward = math.cos(pole_angle)

        # Apply a significant penalty for failure to discourage falling
        if terminated:
            reward = -10.0

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

