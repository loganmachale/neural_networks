import torch
import torch.nn as nn

class LTC_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, ode_time_steps=5):
        """
        Initializes the LTC cell.
        
        Args:
            input_size (int): The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
            ode_time_steps (int): The number of ODE solver steps per forward pass
        """
        super(LTC_Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ode_time_steps = ode_time_steps # Store the time steps

        # Sensory and recurrent weights
        self.sensory_mu = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.sensory_sigma = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.sensory_W = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.sensory_erev = nn.Parameter(torch.Tensor(input_size, hidden_size))

        self.mu = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.sigma = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.erev = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        # Leak and membrane properties
        self.gleak = nn.Parameter(torch.Tensor(1, hidden_size))
        self.vleak = nn.Parameter(torch.Tensor(1, hidden_size))
        self.cm = nn.Parameter(torch.Tensor(1, hidden_size))

        self.init_parameters()

    def init_parameters(self):
        """Initialize parameters with Xavier uniform distribution for better training."""
        nn.init.xavier_uniform_(self.sensory_mu)
        nn.init.xavier_uniform_(self.sensory_sigma)
        nn.init.xavier_uniform_(self.sensory_W)
        nn.init.xavier_uniform_(self.sensory_erev)
        nn.init.xavier_uniform_(self.mu)
        nn.init.xavier_uniform_(self.sigma)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.erev)
        nn.init.xavier_uniform_(self.gleak)
        nn.init.xavier_uniform_(self.vleak)
        nn.init.xavier_uniform_(self.cm)

    def _sigmoid(self, v_pre, mu, sigma):
        """Sigmoid activation function."""
        v_pre = v_pre.unsqueeze(-1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def forward(self, input, hx):
        """
        Performs a forward pass of the LTC cell.
        
        Args:
            input (Tensor): The input tensor of shape (batch_size, input_size)
            hx (Tensor): The hidden state tensor of shape (batch_size, hidden_size)
        
        Returns:
            Tensor: The next hidden state tensor of shape (batch_size, hidden_size)
        """
        # Ensure input and hidden state are 2D
        if input.dim() == 1:
            input = input.unsqueeze(0)
        if hx.dim() == 1:
            hx = hx.unsqueeze(0)

        # Constrain parameters to ensure stability during training
        cm = torch.clamp(self.cm, min=1e-5)
        gleak = torch.clamp(self.gleak, min=1e-5)
        W = torch.clamp(self.W, min=1e-5)
        sensory_W = torch.clamp(self.sensory_W, min=1e-5)

        # Pre-compute sensory activations
        sensory_w_activation = sensory_W * self._sigmoid(input, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # ODE integration loop
        v_pre = hx
        for _ in range(self.ode_time_steps):
            w_activation = W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev

            w_numerator_recurrent = torch.sum(rev_activation, dim=1)
            w_denominator_recurrent = torch.sum(w_activation, dim=1)

            w_numerator = w_numerator_sensory + w_numerator_recurrent
            w_denominator = w_denominator_sensory + w_denominator_recurrent

            numerator = (cm * v_pre) + (gleak * self.vleak) + w_numerator
            denominator = cm + gleak + w_denominator
            
            # Update state using the semi-implicit Euler method
            v_pre = numerator / denominator

        return v_pre

