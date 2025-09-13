import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class ScheduledCartPoleEnv(gym.Env):
    """
    A custom Cart-Pole environment with an expanded track and improved rewards.
    The agent must balance the pole while navigating to a scheduled target.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- CRITICAL FIX: Increase max steps to allow for long-term planning ---
        self.max_episode_steps = 1000 

        # Initialize the standard cart-pole environment with the new step limit
        self.cart_pole = gym.make(
            'CartPole-v1', 
            render_mode=self.render_mode, 
            max_episode_steps=self.max_episode_steps
        )
        self.unwrapped_cart_pole = self.cart_pole.unwrapped
        
        # --- CRITICAL FIX: Widen the track to give the agent more room ---
        self.unwrapped_cart_pole.x_threshold = 4.8 
        
        # Augment the observation space to match the new threshold
        # [cart_pos, cart_vel, pole_angle, pole_vel, target_pos, time_to_target]
        self.observation_space = spaces.Box(
            low=np.array([-self.unwrapped_cart_pole.x_threshold, -np.inf, -0.418, -np.inf, -self.unwrapped_cart_pole.x_threshold, 0]),
            high=np.array([self.unwrapped_cart_pole.x_threshold, np.inf, 0.418, np.inf, self.unwrapped_cart_pole.x_threshold, np.inf]),
            dtype=np.float32
        )
        self.action_space = self.cart_pole.action_space

        # Environment parameters
        self.target_position = 0.0
        self.time_to_target = 0
        self.min_schedule_time = 100 # Increased min time for better planning
        self.max_schedule_time = 400 # Increased max time
        self.current_step = 0

    def _schedule_new_target(self):
        """Schedules a new target position and activation time."""
        self.target_position = self.np_random.uniform(-3.5, 3.5) # Use a slightly smaller range than the threshold
        self.time_to_target = self.np_random.integers(self.min_schedule_time, self.max_schedule_time)

    def _get_obs(self, cart_pole_obs):
        """Construct the augmented observation."""
        time_remaining = max(0, self.time_to_target - self.current_step)
        return np.append(cart_pole_obs, [self.target_position, time_remaining]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cart_pole_obs, info = self.cart_pole.reset(seed=seed)
        self.current_step = 0
        self._schedule_new_target()
        return self._get_obs(cart_pole_obs), info

    def step(self, action):
        cart_pole_obs, reward, terminated, truncated, info = self.cart_pole.step(action)
        self.current_step += 1
        done = terminated or truncated

        # --- Improved Reward Shaping ---
        # 1. Strong penalty for failing to balance
        if terminated:
            return self._get_obs(cart_pole_obs), -10.0, done, False, info

        # 2. Base reward for surviving
        balance_reward = 0.1 

        # 3. Continuous penalty for distance to target
        distance_to_target = abs(cart_pole_obs[0] - self.target_position)
        distance_penalty = -0.1 * distance_to_target 

        # 4. Large bonus for being at the target when it's active
        planning_bonus = 0.0
        time_remaining = self.time_to_target - self.current_step
        if time_remaining <= 0:
            # Generous reward shaping for being close
            planning_bonus = 5.0 * np.exp(- (distance_to_target**2) / 0.25)
        
        # Total reward
        reward = balance_reward + distance_penalty + planning_bonus

        return self._get_obs(cart_pole_obs), reward, done, False, info

    def render(self):
        if self.render_mode is None:
            return None

        render_output = self.cart_pole.render()
        
        # For 'human' mode, rendering happens in the cart_pole env.
        # For 'rgb_array', we just need to add our custom elements.
        # This logic is now handled in the visualization script.
        return render_output


    def close(self):
        self.cart_pole.close()

