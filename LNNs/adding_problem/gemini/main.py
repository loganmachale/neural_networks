import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import json
import os
from html import escape

# Integrated torchdiffeq for advanced ODE solving.
try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    print("Error: torchdiffeq library not found. Please install it (`pip install torchdiffeq`).")
    exit()

class CustomLNN(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, p_excitatory=0.8):
        super(CustomLNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.num_hidden_layers = len(hidden_sizes)

        self.log_tau = nn.ParameterList([nn.Parameter(torch.full((size,), np.log(5.0))) for size in hidden_sizes])
        self.log_Cm = nn.ParameterList([nn.Parameter(torch.full((size,), np.log(1.0))) for size in hidden_sizes])
        self.x_leak = nn.ParameterList([nn.Parameter(torch.full((size,), 0.0)) for size in hidden_sizes])

        all_dims = [self.input_dim] + hidden_sizes
        self.W_abs = nn.ParameterList()
        self.W_polarity = []
        self.E_rev = []

        for i in range(self.num_hidden_layers):
            dim_in = all_dims[i]
            dim_out = all_dims[i+1]
            polarity = torch.where(torch.rand(dim_out, dim_in) < p_excitatory, 1.0, -1.0)
            self.W_polarity.append(polarity)
            self.E_rev.append(torch.where(polarity > 0, 1.0, -1.0))
            self.W_abs.append(nn.Parameter(torch.abs(torch.randn(dim_out, dim_in) * 0.1)))

        for size in hidden_sizes:
            polarity = torch.where(torch.rand(size, size) < p_excitatory, 1.0, -1.0)
            self.W_polarity.append(polarity)
            self.E_rev.append(torch.where(polarity > 0, 1.0, -1.0))
            self.W_abs.append(nn.Parameter(torch.abs(torch.randn(size, size) * 0.1)))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_dim)

    def _ode_func_list(self, t, A_list, u):
        dAs = []
        prev_layer_output = u
        for l in range(self.num_hidden_layers):
            A = A_list[l]
            W_in = torch.abs(self.W_abs[l]) * self.W_polarity[l].to(A.device)
            E_in = self.E_rev[l].to(A.device)
            rec_idx = self.num_hidden_layers + l
            W_rec = torch.abs(self.W_abs[rec_idx]) * self.W_polarity[rec_idx].to(A.device)
            E_rec = self.E_rev[rec_idx].to(A.device)
            inv_tau = 1.0 / torch.exp(self.log_tau[l])
            inv_Cm = 1.0 / torch.exp(self.log_Cm[l])
            tanh_A = torch.tanh(A)
            coupling_in = F.linear(prev_layer_output, W_in)
            coupling_E_in = F.linear(prev_layer_output, W_in * E_in)
            coupling_rec = F.linear(tanh_A, W_rec)
            coupling_E_rec = F.linear(tanh_A, W_rec * E_rec)
            total_coupling = coupling_in + coupling_rec
            total_coupling_E = coupling_E_in + coupling_E_rec
            term1 = -(inv_tau + total_coupling * inv_Cm) * A
            term2 = (self.x_leak[l] * inv_tau) + (total_coupling_E * inv_Cm)
            dA = term1 + term2
            dAs.append(dA)
            prev_layer_output = tanh_A
        return dAs

    def _ode_func_flat(self, t, h, u):
        A_list = list(torch.split(h, self.hidden_sizes, dim=-1))
        dA_list = self._ode_func_list(t, A_list, u)
        return torch.cat(dA_list, dim=-1)

    def forward(self, x, integration_time_per_step):
        batch_size, T, _ = x.shape
        A_list = [torch.zeros(batch_size, size, device=x.device) for size in self.hidden_sizes]
        A_flat = torch.cat(A_list, dim=-1)
        t_span_step = torch.tensor([0.0, integration_time_per_step], device=x.device)
        adjoint_params = tuple(self.parameters())

        for t in range(T):
            u = x[:, t, :]
            ode_func_with_input = lambda time, h: self._ode_func_flat(time, h, u)
            solution = odeint(
                ode_func_with_input, A_flat, t_span_step, method='dopri5', rtol=1e-4, atol=1e-5,
                adjoint_params=adjoint_params
            )
            A_flat = solution[-1]

        final_A_list = torch.split(A_flat, self.hidden_sizes, dim=-1)
        final_A_last_layer = final_A_list[-1]
        output = self.output_layer(torch.tanh(final_A_last_layer))
        return output

def generate_adding_data(batch_size, T, device):
    values = torch.rand(batch_size, T)
    masks = torch.zeros(batch_size, T)
    t1 = torch.randint(0, T // 2, (batch_size,))
    t2 = torch.randint(T // 2, T, (batch_size,))
    masks[torch.arange(batch_size), t1] = 1.0
    masks[torch.arange(batch_size), t2] = 1.0
    x = torch.stack([values, masks], dim=2)
    y = (values * masks).sum(dim=1, keepdim=True)
    return x.to(device), y.to(device)

# Hyperparameters
T = 50
input_dim = 2
hidden_sizes = [12, 8, 4] 
output_dim = 1
batch_size = 128
epochs = 20
lr = 0.005
INTEGRATION_TIME_PER_STEP = 1.0 

# Define bounds for log_tau and log_Cm to ensure stability
LOG_TAU_CM_MIN = np.log(0.1)
LOG_TAU_CM_MAX = np.log(10.0)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model, optimizer, loss
model = CustomLNN(input_dim, hidden_sizes, output_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.MSELoss()

# Training
losses = []
for epoch in tqdm(range(epochs), desc="Training"):
    x, y = generate_adding_data(batch_size, T, device)
    output = model(x, INTEGRATION_TIME_PER_STEP)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # CHANGE: Clamp the time constant parameters after each step to prevent explosion.
    with torch.no_grad():
        for log_tau_l in model.log_tau:
            log_tau_l.clamp_(LOG_TAU_CM_MIN, LOG_TAU_CM_MAX)
        for log_Cm_l in model.log_Cm:
            log_Cm_l.clamp_(LOG_TAU_CM_MIN, LOG_TAU_CM_MAX)
            
    scheduler.step()
    
    losses.append(loss.item())
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}, LR: {scheduler.get_last_lr()[0]:.5f}")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True); plt.yscale('log')

plt.subplot(1, 2, 2)
with torch.no_grad():
    model.eval()
    test_x, test_y = generate_adding_data(200, T, device)
    preds = model(test_x, INTEGRATION_TIME_PER_STEP)
    
plt.scatter(test_y.cpu().numpy(), preds.cpu().numpy(), alpha=0.6, edgecolors='w', s=50)
plt.plot([0, 2], [0, 2], 'r--', linewidth=2, label='Ideal y=x line')
plt.title("Predictions vs. True Values")
plt.xlabel("True Sum")
plt.ylabel("Predicted Sum")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()