import torch
import torch.nn as nn
import numpy as np

class CfCNet(nn.Module):
    """
    A Closed-form Continuous-time (CfC) Liquid Neural Network.
    This architecture is directly inspired by the LTC research paper.
    """
    def __init__(self, config):
        super(CfCNet, self).__init__()
        self.config = config
        self.input_size = config['environment']['state_size']
        self.hidden_size = config['network']['hidden_size']
        self.motor_size = config['network']['motor_size']
        self.output_size = config['environment']['action_size']

        # Core network components
        self.input_mapping = nn.Linear(self.input_size, self.hidden_size)
        
        # The time-constant network (learns how to react over time)
        self.time_constant_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Softplus() # Ensures time-constants are positive
        )
        
        # The main feed-forward network for state transition
        self.ff_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )
        
        # Output mapping
        self.output_mapping = nn.Sequential(
            nn.Linear(self.hidden_size, self.motor_size),
            nn.ReLU(),
            nn.Linear(self.motor_size, self.output_size),
            nn.Tanh()
        )

    def forward(self, x, hidden_state=None):
        batch_size = x.shape[0]

        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size)

        # Map the physical input to the network's hidden dimension
        mapped_input = self.input_mapping(x)

        # Calculate the time-varying time-constant
        # This is the "liquid" part of the network
        time_constant = self.time_constant_net(mapped_input)
        
        # Calculate the next state using the feed-forward network
        ff_out = self.ff_net(mapped_input)

        # --- Core ODE Solver Step (Closed-form solution) ---
        # This equation is a direct implementation from the research paper [cite: 56]
        next_hidden_state = (
            hidden_state * (1.0 - time_constant) +
            ff_out * time_constant
        )

        # Calculate the final action from the new hidden state
        output = self.output_mapping(next_hidden_state)
        
        return output, next_hidden_state