import numpy as np
import gymnasium as gym
from gymnasium import spaces

class InvertedPendulum(gym.Env):
    """
    Custom Inverted Pendulum environment based on Gymnasium.
    """
    def __init__(self, config):
        super(InvertedPendulum, self).__init__()
        self.config = config['environment']
        self.length = self.config['length']
        self.mass_cart = self.config['mass_cart']
        self.mass_pole = self.config['mass_pole']
        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_moment = self.mass_pole * self.length
        self.gravity = self.config['gravity']
        self.force_mag = self.config['force_mag']
        self.dt = self.config['dt']  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = self.config['theta_threshold_radians']
        self.x_threshold = self.config['x_threshold']

        # Action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
        self.state = None

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag * float(action)
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.pole_moment * theta_dot ** 2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / \
                    (self.length * (4.0 / 3.0 - self.mass_pole * costheta ** 2 / self.total_mass))
        x_acc = temp - self.pole_moment * theta_acc * costheta / self.total_mass

        # Update state using Euler's method
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc
        self.state = (x, x_dot, theta, theta_dot)

        # Check for termination
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        # --- UPDATED REWARD CALCULATION ---
        if not terminated:
            # Reward for being upright
            height_reward = self.config['reward_factor_height'] * np.cos(theta)
            
            # Penalties for movement
            velocity_penalty = self.config['reward_factor_velocity'] * abs(theta_dot)
            cart_pos_penalty = self.config['reward_factor_cart_pos'] * abs(x)
            cart_vel_penalty = self.config['reward_factor_cart_vel'] * abs(x_dot)
            
            reward = height_reward - velocity_penalty - cart_pos_penalty - cart_vel_penalty
        else:
            reward = -10.0 # Punishment for failure

        truncated = False # We don't use truncation in this simple case
        
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # High, low for state initialization
        high = np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        low = -high
        self.state = np.random.uniform(low=low, high=high)
        # Return state and an empty info dictionary, per Gymnasium standard
        return np.array(self.state, dtype=np.float32), {}

    def render(self, mode='human'):
        # This environment is rendered in visualize.py, so this can be a pass-through
        pass

    def close(self):
        pass