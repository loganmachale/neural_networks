import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

class CfC(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CfC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backbone = nn.Linear(input_size + hidden_size, hidden_size * 3)
        self.tanh = nn.Tanh()

    def forward(self, input, hx, dt=1.0):
        x = torch.cat([input, hx], dim=-1)
        bb = self.backbone(x)
        f, g, h = torch.split(bb, self.hidden_size, dim=-1)
        f = self.tanh(f)
        g = self.tanh(g)
        h = self.tanh(h)
        sigma = torch.sigmoid(-f * dt)
        new_hx = sigma * g + (1 - sigma) * h
        return new_hx, new_hx

class CustomLNN(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super(CustomLNN, self).__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim
        all_sizes = hidden_sizes + [output_dim]
        for h in all_sizes:
            self.layers.append(CfC(current_dim, h))
            current_dim = h

    def forward(self, x, fixed_data_mode=False):
        batch_size, T, _ = x.shape
        states = [torch.zeros(batch_size, layer.hidden_size, device=x.device) for layer in self.layers]
        if fixed_data_mode:
            # For fixed data, repeat first input over short T (e.g., 10 steps) to settle dynamics
            T = 10
            x = x[:, 0:1, :].repeat(1, T, 1)  # Constant input repetition
        for t in range(T):
            u = x[:, t, :]
            # Optional: Scale dt based on input (e.g., higher for marked positions in adding)
            # dt = 2.0 if torch.any(u[:, 1] > 0) else 1.0
            dt = 1.0
            for l, layer in enumerate(self.layers):
                input_l = u if l == 0 else states[l-1]
                _, new_state = layer(input_l, states[l], dt=dt)
                states[l] = new_state
        output_A = states[-1]
        return output_A

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
hidden_sizes = [128, 128]  # Increased for better memory on sequences
output_dim = 1
T = 100
batch_size = 128
epochs = 50  # Increased for convergence
lr = 0.05

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Scalers
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

# Model, optimizer, loss
model = CustomLNN(input_dim, hidden_sizes, output_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.MSELoss()

print_interval = max(1, epochs // 10)

# Training (on sequential adding)
losses = []
for epoch in tqdm(range(epochs), desc="Training"):
    x, y = generate_adding_data(batch_size, T, device)
    x_reshaped = x.view(-1, input_dim)
    x_scaled = torch.tensor(x_scaler.fit_transform(x_reshaped.cpu().numpy()), dtype=torch.float32, device=device)
    x_scaled = x_scaled.view(batch_size, T, input_dim)
    y_scaled = torch.tensor(y_scaler.fit_transform(y.cpu().numpy()), dtype=torch.float32, device=device)
    
    output = model(x_scaled)
    loss = criterion(output, y_scaled)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    scheduler.step()
    
    losses.append(loss.item())
    if (epoch + 1) % print_interval == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}, LR: {scheduler.get_last_lr()[0]:.5f}")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.yscale('log')

plt.subplot(1, 2, 2)
with torch.no_grad():
    model.eval()
    test_x, test_y = generate_adding_data(200, T, device)
    test_x_reshaped = test_x.view(-1, input_dim)
    test_x_scaled = torch.tensor(x_scaler.transform(test_x_reshaped.cpu().numpy()), dtype=torch.float32, device=device)
    test_x_scaled = test_x_scaled.view(200, T, input_dim)
    preds_scaled = model(test_x_scaled)
    preds = torch.tensor(y_scaler.inverse_transform(preds_scaled.cpu().numpy()), dtype=torch.float32, device=device)
    
plt.scatter(test_y.cpu().numpy(), preds.cpu().numpy(), alpha=0.6, edgecolors='w', s=50)
plt.plot([0, 2], [0, 2], 'r--', linewidth=2, label='Ideal y=x line')
plt.title("Predictions vs. True Values")
plt.xlabel("True Sum")
plt.ylabel("Predicted Sum")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Demo for Fixed Data (e.g., sum two static vectors without sequence)
with torch.no_grad():
    # Example: Two vectors as batch of 1, T=1 initially
    fixed_x = torch.tensor([[[0.3, 1.0], [0.7, 1.0]]], device=device)  # Values 0.3 and 0.7, both "marked"
    fixed_x_scaled = torch.tensor(x_scaler.transform(fixed_x.view(-1, input_dim).cpu().numpy()), dtype=torch.float32, device=device)
    fixed_x_scaled = fixed_x_scaled.view(1, 2, input_dim)  # Treat as short sequence
    preds_scaled = model(fixed_x_scaled, fixed_data_mode=True)  # Use fixed mode to repeat and settle
    pred_sum = y_scaler.inverse_transform(preds_scaled.cpu().numpy())[0][0]
    print(f"Fixed data predicted sum: {pred_sum:.2f} (true: 1.00)")