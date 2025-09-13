import torch
import torch.optim as optim
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from environment import ScheduledCartPoleEnv
from model import HLTCN

# --- Hyperparameters ---
EPISODES = 2000
GAMMA = 0.99
LEARNING_RATE = 0.001
MAX_STEPS_PER_EPISODE = 500

def main():
    env = ScheduledCartPoleEnv()
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    model = HLTCN(obs_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scores = []
    scores_window = deque(maxlen=100)

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        h_low, h_high = None, None # Reset hidden states at the start of each episode
        
        log_probs = []
        values = []
        rewards = []
        
        for t in range(MAX_STEPS_PER_EPISODE):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            
            # Detach hidden states to prevent gradients from flowing across time steps
            h_low = h_low.detach() if h_low is not None else None
            h_high = h_high.detach() if h_high is not None else None

            action, log_prob, value, h_low_next, h_high_next = model(obs_tensor, h_low, h_high)
            
            h_low, h_high = h_low_next, h_high_next
            
            obs, reward, done, _, _ = env.step(action.item())
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            if done:
                break
        
        current_score = sum(rewards)
        scores.append(current_score)
        scores_window.append(current_score)

        # --- A2C Update ---
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8) # Normalize returns

        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze()
        
        advantages = returns - values
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        
        loss = actor_loss + 0.5 * critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_window):.2f}\tLoss: {loss.item():.4f}")
            
    # Save the model and plot scores
    torch.save(model.state_dict(), 'hltcn_model.pth')
    
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title("Scores per Episode")
    plt.ylabel("Total Reward")
    plt.xlabel("Episode")
    plt.grid(True)
    plt.savefig("training_scores.png")
    plt.show()

if __name__ == "__main__":
    main()

