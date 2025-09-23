import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class CustomLNN(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, tau_mean=1.0, tau_sigma=0.1):
        super(CustomLNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.num_layers = len(hidden_sizes)

        all_sizes = [input_dim] + hidden_sizes

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(all_sizes[i], all_sizes[i+1]))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_dim)

        self.taus = nn.ParameterList([
            nn.Parameter(torch.normal(tau_mean, tau_sigma, (size,))) for size in hidden_sizes
        ])

    def ode_func(self, A_list, u):
        dAs = []
        prev_layer_output = u

        for l in range(self.num_layers):
            A = A_list[l]
            re_tau = 1.0 / (F.relu(self.taus[l]) + 1e-7)
            layer_input = self.layers[l](prev_layer_output)
            dA = re_tau * (-A + layer_input)
            dAs.append(dA)
            prev_layer_output = torch.tanh(A)

        return dAs

    def forward(self, x, dt=0.05):
        batch_size, T, _ = x.shape
        A_list = [torch.zeros(batch_size, size, device=x.device) for size in self.hidden_sizes]
        
        for t in range(T):
            u = x[:, t, :]
            
            k1_list = self.ode_func(A_list, u)
            A_k2 = [A + dt * 0.5 * k1 for A, k1 in zip(A_list, k1_list)]
            k2_list = self.ode_func(A_k2, u)
            A_k3 = [A + dt * 0.5 * k2 for A, k2 in zip(A_list, k2_list)]
            k3_list = self.ode_func(A_k3, u)
            A_k4 = [A + dt * k3 for A, k3 in zip(A_list, k3_list)]
            k4_list = self.ode_func(A_k4, u)

            A_list = [
                A + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                for A, k1, k2, k3, k4 in zip(A_list, k1_list, k2_list, k3_list, k4_list)
            ]
            
        final_A = A_list[-1]
        output = self.output_layer(torch.tanh(final_A))
        return output

def generate_adding_data(batch_size, T, device):
    values = torch.rand(batch_size, T, device=device)
    
    # CHANGE: The mask is now encoded as [-1, +1] instead of [0, 1].
    # -1 provides an active inhibitory signal for the "ignore" timesteps.
    masks = torch.full((batch_size, T), -1.0, device=device)
    
    t1 = torch.randint(0, T // 2, (batch_size,), device=device)
    t2 = torch.randint(T // 2, T, (batch_size,), device=device)
    
    # Place the +1 "attend" signals at the correct locations.
    masks.scatter_(1, t1.unsqueeze(1), 1.0)
    masks.scatter_(1, t2.unsqueeze(1), 1.0)
    
    x = torch.stack([values, masks], dim=2)
    
    # The target y is still the sum of values where the original mask was 1.
    # We can calculate this by finding where the new mask is > 0.
    y = (values * (masks > 0)).sum(dim=1, keepdim=True)
    return x, y

# Hyperparameters
input_dim = 2
hidden_sizes = [32, 32]
output_dim = 1
T = 100
batch_size = 128
epochs = 20
lr = 0.005
dt = 0.05
lr_decay_step = 5
lr_decay_gamma = 0.5

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model, optimizer, loss
model = CustomLNN(input_dim, hidden_sizes, output_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
criterion = nn.MSELoss()

print_interval = max(1, epochs // 20)

# Training
losses = []
for epoch in tqdm(range(epochs), desc="Training"):
    x, y = generate_adding_data(batch_size, T, device)
    output = model(x, dt=dt)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    
    losses.append(loss.item())
    if (epoch + 1) % print_interval == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}, LR: {scheduler.get_last_lr()[0]:.5f}")

# Results/Analysis Visualization
plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.yscale('log')

# Test predictions vs true
plt.subplot(1, 2, 2)
with torch.no_grad():
    model.eval()
    test_x, test_y = generate_adding_data(200, T, device)
    preds = model(test_x, dt=dt)
    
plt.scatter(test_y.cpu().numpy(), preds.cpu().numpy(), alpha=0.6, edgecolors='w', s=50)
plt.plot([0, 2], [0, 2], 'r--', linewidth=2, label='Ideal y=x line')
plt.title("Predictions vs. True Values")
plt.xlabel("True Sum")
plt.ylabel("Predicted Sum")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()