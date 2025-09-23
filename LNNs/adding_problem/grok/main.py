import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class CustomLNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, tau1=1.0, sigma=0.1):
        super(CustomLNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
       
        self.W_input = nn.Parameter(torch.empty(hidden_size, input_dim))
        self.W = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.readout = nn.Linear(hidden_size, output_dim)
       
        nn.init.xavier_uniform_(self.W_input)
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
       
        self.tau = nn.Parameter(torch.normal(tau1, sigma, (hidden_size,)))

    def ode_func(self, state, u):
        batch_size = state.shape[0]
        A = state
        tau = F.relu(self.tau).expand(batch_size, -1)
        sig_A = torch.sigmoid(A)
        intra_term = sig_A @ self.W.T
        sig_u = torch.sigmoid(u)
        inter_term = sig_u @ self.W_input.T
        re_tau = 1.0 / (tau + 1e-5)
        dA = re_tau * (intra_term + inter_term + self.bias)
        return dA

    def forward(self, x, dt=0.05):
        batch_size, T, _ = x.shape
       
        initial_A = torch.zeros(batch_size, self.hidden_size, device=x.device)
        state = initial_A
       
        steps_per_input = int(1.0 / dt)
        for t in range(T):
            u = x[:, t, :]
            for _ in range(steps_per_input):
                k1 = self.ode_func(state, u)
                k2 = self.ode_func(state + dt * 0.5 * k1, u)
                k3 = self.ode_func(state + dt * 0.5 * k2, u)
                k4 = self.ode_func(state + dt * k3, u)
                state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
       
        output = self.readout(state)
        return output

def generate_adding_data(batch_size, T, device):
    values = torch.rand(batch_size, T, device=device)
    masks = torch.zeros(batch_size, T, device=device)
    t1 = torch.randint(0, T // 2, (batch_size,), device=device)
    t2 = torch.randint(T // 2, T, (batch_size,), device=device)
    masks.scatter_(1, t1.unsqueeze(1), 1.0)
    masks.scatter_(1, t2.unsqueeze(1), 1.0)
   
    x = torch.stack([values, masks], dim=2)
    y = (values * masks).sum(dim=1, keepdim=True)
    return x, y

# Hyperparameters
input_dim = 2
hidden_size = 32
output_dim = 1
T = 50
batch_size = 128
epochs = 10
lr = 0.001
dt = 0.05
tau1 = 1.0
sigma = 0.1
lr_decay_step = 50
lr_decay_gamma = 0.5

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model, optimizer, loss
model = CustomLNN(input_dim, hidden_size, output_dim, tau1=tau1, sigma=sigma)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
criterion = nn.MSELoss()
print_interval = max(1, epochs // 10)

# Training
losses = []
for epoch in tqdm(range(epochs), desc="Training"):
    x, y = generate_adding_data(batch_size, T, device)
    output = model(x, dt=dt)
    loss = criterion(output, y)
   
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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