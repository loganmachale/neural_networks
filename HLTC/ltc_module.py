import torch
import torch.nn as nn

class LTC_Cell(nn.Module):
    """A Liquid Time-Constant (LTC) cell."""
    def __init__(self, input_size, hidden_size, time_constant_log_init):
        super(LTC_Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Learnable parameters for the ODE
        self.w_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        
        # Time constant parameter (log space for stability)
        self.time_constant_log = nn.Parameter(torch.Tensor(hidden_size))
        
        # Initialize parameters
        self._initialize_weights(time_constant_log_init)

    def _initialize_weights(self, time_constant_log_init):
        nn.init.xavier_uniform_(self.w_ih)
        nn.init.orthogonal_(self.w_hh)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.time_constant_log, time_constant_log_init)

    def forward(self, x, hidden_state, dt=1.0):
        """
        Forward pass using a fixed-step solver (Fused Solver from the paper).
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)
            hidden_state (Tensor): Hidden state from the previous time step
                                  of shape (batch_size, hidden_size)
            dt (float): Time step for the solver.
        """
        if hidden_state is None:
            hidden_state = torch.zeros(x.size(0), self.hidden_size, device=x.device)

        # Compute the non-linear gate f(x, h, t, theta)
        gate_input = torch.matmul(x, self.w_ih.t()) + torch.matmul(hidden_state, self.w_hh.t()) + self.bias
        f_gate = torch.sigmoid(gate_input)

        # Compute the inverse of the time constant
        inv_tau = torch.exp(-self.time_constant_log)
        
        # The ODE is dx/dt = - (1/tau + f) * x + f * A
        # The paper simplifies by setting A=1 (implicitly via bias in the gate)
        # We use the fused solver from Eq. 3 of the LTC paper.
        
        numerator = hidden_state + dt * f_gate
        denominator = 1 + dt * (inv_tau + f_gate)
        
        next_hidden_state = numerator / denominator
        
        return next_hidden_state

