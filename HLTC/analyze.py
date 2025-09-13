import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

from environment import ScheduledCartPoleEnv
from model import HLTCN

def run_analysis():
    env = ScheduledCartPoleEnv(render_mode='human')
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = HLTCN(obs_size, action_size)
    try:
        model.load_state_dict(torch.load('hltcn_model.pth'))
    except FileNotFoundError:
        print("Model file 'hltcn_model.pth' not found. Please run train.py first.")
        return
        
    model.eval()

    obs, _ = env.reset()
    h_low, h_high = None, None
    
    h_low_history = []
    h_high_history = []
    cart_pos_history = []
    target_pos_history = []
    pole_angle_history = []

    for i in range(500):
        env.render()
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        
        with torch.no_grad():
            action, _, _, h_low_next, h_high_next = model(obs_tensor, h_low, h_high)
        
        h_low, h_high = h_low_next, h_high_next
        
        # Record data for plotting
        h_low_history.append(h_low.squeeze().numpy())
        h_high_history.append(h_high.squeeze().numpy())
        cart_pos_history.append(obs[0])
        target_pos_history.append(obs[4])
        pole_angle_history.append(obs[2])
        
        obs, _, done, _, _ = env.step(action.item())
        if done:
            print(f"Episode finished after {i+1} steps.")
            break
        
        time.sleep(0.01) # Slow down rendering a bit
            
    env.close()

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Behavior
    axs[0].plot(cart_pos_history, label='Cart Position', color='blue')
    axs[0].plot(target_pos_history, label='Target Position', color='green', linestyle='--')
    axs[0].set_ylabel('Position')
    axs[0].set_title('Cart and Target Positions over Time')
    axs[0].legend()
    ax0_twin = axs[0].twinx()
    ax0_twin.plot(pole_angle_history, label='Pole Angle (rad)', color='red', alpha=0.5)
    ax0_twin.set_ylabel('Pole Angle', color='red')
    ax0_twin.legend(loc='upper right')
    
    # Plot 2: Low-level state activity
    if h_low_history:
        pca = PCA(n_components=2)
        h_low_pca = pca.fit_transform(np.array(h_low_history))
        axs[1].plot(h_low_pca[:, 0], label='PC1')
        axs[1].plot(h_low_pca[:, 1], label='PC2')
        axs[1].set_ylabel('Activation')
        axs[1].set_title('Low-Level (Fast) Module State (PCA) - Should be noisy/reactive')
        axs[1].legend()

    # Plot 3: High-level state activity
    if h_high_history:
        pca = PCA(n_components=2)
        h_high_pca = pca.fit_transform(np.array(h_high_history))
        axs[2].plot(h_high_pca[:, 0], label='PC1')
        axs[2].plot(h_high_pca[:, 1], label='PC2')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Activation')
        axs[2].set_title('High-Level (Slow) Module State (PCA) - Should be smooth/stable')
        axs[2].legend()
    
    plt.tight_layout()
    plt.savefig("analysis_plots.png")
    plt.show()

if __name__ == "__main__":
    run_analysis()

