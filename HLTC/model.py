import torch
import torch.nn as nn
from torch.distributions import Categorical
from ltc_module import LTC_Cell

class HLTCN(nn.Module):
    """
    Hierarchical Liquid Time-Constant Network (HLTCN)
    Combines a fast low-level module for control and a slow
    high-level module for planning.
    """
    def __init__(self, obs_size, action_size, low_level_size=32, high_level_size=16):
        super(HLTCN, self).__init__()

        # --- Hyperparameters ---
        self.low_level_steps = 5  # T: Number of fast steps per slow step

        # --- Modules ---
        # Fast module: small tau -> large inv_tau -> quick reaction
        # We initialize log(tau) = -2.0 => tau ~ 0.13
        self.ltc_low = LTC_Cell(obs_size + high_level_size, low_level_size, time_constant_log_init=-2.0)
        
        # Slow module: large tau -> small inv_tau -> slow integration
        # We initialize log(tau) = 1.0 => tau ~ 2.7
        self.ltc_high = LTC_Cell(low_level_size, high_level_size, time_constant_log_init=1.0)
        
        # Actor head (from low-level state)
        self.actor_head = nn.Linear(low_level_size, action_size)
        
        # Critic head (from high-level state)
        self.critic_head = nn.Linear(high_level_size, 1)

    def forward(self, obs, h_low=None, h_high=None):
        """
        A single forward pass representing one "segment" of HRM.
        
        Args:
            obs (Tensor): Current environment observation
            h_low (Tensor): Previous low-level hidden state
            h_high (Tensor): Previous high-level hidden state
        """
        if h_high is None:
             h_high = torch.zeros(obs.size(0), self.ltc_high.hidden_size, device=obs.device)

        # --- Low-level (fast) loop ---
        for _ in range(self.low_level_steps):
            # The high-level state acts as a constant context for the low-level loop
            combined_input = torch.cat([obs, h_high], dim=-1)
            h_low = self.ltc_low(combined_input, h_low)

        # --- High-level (slow) update ---
        h_high = self.ltc_high(h_low, h_high)
        
        # --- Actor and Critic Outputs ---
        action_logits = self.actor_head(h_low)
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        
        value_estimate = self.critic_head(h_high)
        
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob, value_estimate, h_low, h_high

