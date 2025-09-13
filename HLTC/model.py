import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ltc_module import LTC_Cell

class HLTCN(nn.Module):
    def __init__(self, obs_size, action_size, h_units=32, l_units=32, t_low=5, t_high=20):
        super(HLTCN, self).__init__()
        
        # --- FIX: Store hidden layer sizes as attributes ---
        self.h_high_size = h_units
        self.h_low_size = l_units
        # ---------------------------------------------------

        # The fast, low-level module for reactive control
        self.low_level = LTC_Cell(obs_size, self.h_low_size, t_low)

        # The slow, high-level module for strategic planning
        self.high_level = LTC_Cell(self.h_low_size, self.h_high_size, t_high)

        # Actor: Policy head that decides on an action
        self.actor = nn.Linear(self.h_high_size, action_size)
        
        # Critic: Value head that estimates the quality of a state
        self.critic = nn.Linear(self.h_high_size, 1)

    def init_hidden(self, batch_size=1):
        """Initializes the hidden states for both modules."""
        h_low = torch.zeros(batch_size, self.h_low_size)
        h_high = torch.zeros(batch_size, self.h_high_size)
        return h_low, h_high

    def forward(self, obs, h_low, h_high):
        # The low-level module processes the raw environment observation
        h_low_next = self.low_level(obs, h_low)

        # The high-level module processes the output of the low-level module
        h_high_next = self.high_level(h_low_next, h_high)

        # The critic estimates the value of the current state based on the high-level plan
        value = self.critic(h_high_next)
        
        # The actor determines the action probabilities based on the high-level plan
        action_logits = self.actor(h_high_next)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        
        return action, log_prob, value, h_low_next, h_high_next

