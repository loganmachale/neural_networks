import torch
import torch.optim as optim
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from balancing_env import BalancingCartPoleEnv # Import the new environment
from model import HLTCN

# --- Hyperparameters ---
EPISODES = 1000 # This task is simpler, so fewer episodes may be needed
GAMMA = 0.98
LEARNING_RATE = 0.0005
SEGMENT_LENGTH = 12

def main():
    env = BalancingCartPoleEnv() # Use the new balancing-focused environment
    obs_size = env.observation_space.shape[0] # Will be 4
    action_size = env.action_space.n
    
    model = HLTCN(obs_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scores = []
    scores_window = deque(maxlen=100)
    
    progress_bar = tqdm(range(1, EPISODES + 1), desc="Training Balancer", unit="episode")

    for episode in progress_bar:
        obs, _ = env.reset()
        h_low, h_high = model.init_hidden(batch_size=1)
        
        episode_rewards = []
        done = False
        
        while not done:
            log_probs, values, rewards = [], [], []

            # --- Collect a segment of experience ---
            for _ in range(SEGMENT_LENGTH):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                
                if h_low is not None: h_low = h_low.detach()
                if h_high is not None: h_high = h_high.detach()

                action, log_prob, value, h_low_next, h_high_next = model(obs_tensor, h_low, h_high)
                
                h_low, h_high = h_low_next, h_high_next
                obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                episode_rewards.append(reward)
                
                if done:
                    break
            
            # --- Deep Supervision Update ---
            R = 0
            if not done:
                with torch.no_grad():
                    _, _, last_value, _, _ = model(torch.from_numpy(obs).float().unsqueeze(0), h_low, h_high)
                    R = last_value.item()

            returns = []
            for r in reversed(rewards):
                R = r + GAMMA * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns, dtype=torch.float32).view(-1, 1)
            log_probs = torch.cat(log_probs)
            values = torch.cat(values)
            
            advantages = returns - values
            
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        current_score = sum(episode_rewards)
        scores.append(current_score)
        scores_window.append(current_score)

        if episode % 10 == 0:
            progress_bar.set_postfix({
                "Avg Score": f"{np.mean(scores_window):.2f}",
                "Loss": f"{loss.item():.4f}"
            })
            
    # --- Save the trained balancing model to a new file ---
    torch.save(model.state_dict(), 'balancing_model.pth')
    print("\nBalancing model saved to balancing_model.pth")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title("Balancing Training Scores per Episode")
    plt.ylabel("Total Reward")
    plt.xlabel("Episode")
    moving_avg = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
    plt.plot(moving_avg, color='red', linewidth=2, label='100-episode MA')
    plt.grid(True)
    plt.legend()
    plt.savefig("balancing_training_scores.png")
    plt.show()

if __name__ == "__main__":
    main()
