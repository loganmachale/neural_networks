import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class ScheduledCartPoleEnv(gym.Env):
    """
    A custom Cart-Pole environment where the agent must balance the pole
    while navigating to a target zone that appears at a future time.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 50}

    def __init__(self, render_mode=None):
        super().__init__()
        # Initialize the standard cart-pole environment
        self.cart_pole = gym.make('CartPole-v1')

        # Augment the observation space
        # [cart_pos, cart_vel, pole_angle, pole_vel, target_pos, time_to_target]
        self.observation_space = spaces.Box(
            low=np.array([-4.8, -np.inf, -0.418, -np.inf, -4.8, 0]),
            high=np.array([4.8, np.inf, 0.418, np.inf, 4.8, np.inf]),
            dtype=np.float32
        )
        self.action_space = self.cart_pole.action_space

        # Environment parameters
        self.target_position = 0.0
        self.time_to_target = 0
        self.min_schedule_time = 50  # Min steps until target appears
        self.max_schedule_time = 200 # Max steps until target appears
        self.current_step = 0

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def _schedule_new_target(self):
        """Schedules a new target position and activation time."""
        self.target_position = self.np_random.uniform(-2.4, 2.4)
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

        # --- Custom Reward Shaping ---
        # 1. Base balancing reward (is always 1 if not terminated)
        balance_reward = 1.0

        # 2. Planning reward: bonus for being near the target at the right time
        planning_reward = 0.0
        time_remaining = self.time_to_target - self.current_step
        if time_remaining <= 0:
            distance_to_target = abs(cart_pole_obs[0] - self.target_position)
            # Use a sharp reward function like a Gaussian
            planning_reward = 2.0 * np.exp(- (distance_to_target**2) / 0.1)

        # 3. Penalize large actions to encourage smooth control
        action_penalty = -0.01 * (action - 0.5)**2

        reward = balance_reward + planning_reward + action_penalty

        # If the pole falls, give a large penalty
        if terminated:
            reward = -10.0

        return self._get_obs(cart_pole_obs), reward, done, False, info

    def render(self):
        if self.render_mode is None:
            return

        # Use the underlying cart-pole's render method
        render_result = self.cart_pole.render()
        if self.screen is None and self.render_mode == 'human':
             # The cart_pole render might return the screen
            if render_result is not None and isinstance(render_result, pygame.Surface):
                 self.screen = render_result
            else: # Fallback if it doesn't return screen
                pygame.init()
                self.screen = pygame.display.set_mode((600, 400))


        # Draw the target zone
        if self.clock is None:
            self.clock = pygame.time.Clock()

        screen_width = 600
        world_width = 4.8
        scale = screen_width / world_width
        target_x = self.target_position * scale + screen_width / 2.0

        time_remaining = self.time_to_target - self.current_step
        if time_remaining > 0:
            # Draw a "ghost" of the target
            color = (150, 150, 255) # Semi-transparent blue
            surface = pygame.Surface((20, 200), pygame.SRCALPHA)
            surface.fill((*color, 128))
            self.screen.blit(surface, (target_x - 10, 100))
        else:
            # Draw the active target
            color = (50, 200, 50) # Green
            target_rect = pygame.Rect(target_x - 10, 100, 20, 200)
            pygame.draw.rect(self.screen, color, target_rect)
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])


    def close(self):
        self.cart_pole.close()
        if self.screen:
            pygame.quit()
            self.screen = None

