import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import yaml
import json
from tqdm import tqdm

from network import CfCNet # Updated import
from environment import InvertedPendulum

class ActorCritic(nn.Module):
    """
    An Actor-Critic wrapper for the CfCNet.
    """
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.hidden_size = config['network']['hidden_size']
        self.output_size = config['environment']['action_size']

        # The core network is now the CfCNet
        self.cfc = CfCNet(config)
        
        # Critic head predicts state value from the CfC's hidden state
        self.critic_head = nn.Linear(self.hidden_size, 1)
        # Learnable parameter for action standard deviation
        self.actor_log_std = nn.Parameter(torch.zeros(1, self.output_size))

    def forward(self, x, hidden_state=None):
        action_mean, new_hidden_state = self.cfc(x, hidden_state)
        
        value = self.critic_head(new_hidden_state)
        action_std = torch.exp(self.actor_log_std)
        
        return action_mean, action_std, value, new_hidden_state

def train():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    env = InvertedPendulum(config)
    model = ActorCritic(config) # Use the new ActorCritic
    
    training_config = config['training']
    learning_rate = training_config['learning_rate']
    
    training_logs = {"losses": [], "rewards": []}
    print("Starting PPO training with CfC Network...")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    num_episodes = training_config['num_episodes']
    gamma = training_config['gamma']
    ppo_epochs = training_config['ppo_epochs']
    ppo_clip = training_config['ppo_clip']

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        memory = []
        state, _ = env.reset()
        hidden_state = None
        episode_rewards = 0

        for step in range(training_config['max_steps_per_episode']):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            with torch.no_grad():
                action_mean, action_std, value, new_hidden_state = model(state_tensor, hidden_state)
            
            action_dist = Normal(action_mean, action_std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            action_np = action.detach().numpy().flatten()
            next_state, reward, done, _, _ = env.step(action_np)
            
            memory.append((state, action, log_prob, reward, done, value))
            state = next_state
            hidden_state = new_hidden_state
            episode_rewards += reward
            
            if done: break
        
        training_logs["rewards"].append(episode_rewards)

        if not memory: continue

        states, actions, old_log_probs, rewards, dones, values = zip(*memory)
        
        discounted_rewards = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done: R = 0
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.cat(actions)
        old_log_probs_t = torch.cat(old_log_probs).detach()
        rewards_t = torch.tensor(discounted_rewards, dtype=torch.float32)
        values_t = torch.cat(values).squeeze(-1).detach()

        advantages = rewards_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(ppo_epochs):
            action_means, action_stds, current_values, _ = model(states_t, None)
            current_values = current_values.squeeze()
            
            dist = Normal(action_means, action_stds)
            current_log_probs = dist.log_prob(actions_t)
            dist_entropy = dist.entropy()

            ratios = torch.exp(current_log_probs - old_log_probs_t)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - ppo_clip, 1 + ppo_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = mse_loss(current_values, rewards_t)
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        training_logs["losses"].append(loss.item())

    save_path = training_config['model_save_path']
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    with open(training_config['log_save_path'], 'w') as f:
        json.dump(training_logs, f, indent=4)

    print("Training complete.")

if __name__ == '__main__':
    train()