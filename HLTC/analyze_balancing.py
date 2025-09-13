import torch
import numpy as np
import pygame
import time

from balancing_env import BalancingCartPoleEnv # Use the new environment
from model import HLTCN

# --- Visualization Constants ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 480
SIM_PANEL_WIDTH = 600
NET_PANEL_WIDTH = 600
BACKGROUND_COLOR = (25, 35, 45)
NEURON_RADIUS = 10
LAYER_SPACING = 150
NEURON_SPACING = 25

# --- Drawing Function for the Network ---
def draw_network(surface, h_low, h_high, obs, h_low_size, h_high_size, obs_size):
    """Draws the network activations on the given Pygame surface."""
    surface.fill(BACKGROUND_COLOR)
    
    # Normalize activations for color mapping
    # Use tanh to squash values into a [-1, 1] range for better visualization
    if h_low is not None:
        h_low_norm = torch.tanh(h_low).squeeze().numpy()
    else:
        h_low_norm = np.zeros(h_low_size)

    if h_high is not None:
        h_high_norm = torch.tanh(h_high).squeeze().numpy()
    else:
        h_high_norm = np.zeros(h_high_size)
        
    # Normalize observations. The ranges are roughly [-2.4, 2.4], [-inf, inf], [-0.2, 0.2], [-inf, inf]
    # We can create a simple normalization for visualization purposes.
    obs_norm = np.clip(obs / np.array([2.4, 2.0, 0.2, 2.0]), -1, 1)

    # Helper to draw a single layer
    def draw_layer(neurons, x_pos, title):
        font = pygame.font.SysFont(None, 24)
        text = font.render(title, True, (200, 200, 200))
        surface.blit(text, (x_pos - text.get_width() // 2, 20))

        total_height = (len(neurons) - 1) * NEURON_SPACING
        start_y = (SCREEN_HEIGHT - total_height) // 2

        for i, activation in enumerate(neurons):
            y_pos = start_y + i * NEURON_SPACING
            # Map activation from [-1, 1] to color [0, 255]
            color_val = int((activation + 1) / 2 * 255)
            color = (color_val, color_val, color_val) # Grayscale
            pygame.draw.circle(surface, color, (x_pos, y_pos), NEURON_RADIUS)
            pygame.draw.circle(surface, (100, 100, 100), (x_pos, y_pos), NEURON_RADIUS, 1) # Border

    # Draw each layer
    draw_layer(obs_norm, 100, "Inputs")
    draw_layer(h_high_norm, 100 + LAYER_SPACING, "High-Level (Slow)")
    draw_layer(h_low_norm, 100 + 2 * LAYER_SPACING, "Low-Level (Fast)")


def run_visualization():
    # --- Pygame Setup ---
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("HLTCN Balancing Analysis")
    sim_surface = pygame.Surface((SIM_PANEL_WIDTH, SCREEN_HEIGHT))
    net_surface = pygame.Surface((NET_PANEL_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    # --- Environment and Model Setup ---
    env = BalancingCartPoleEnv(render_mode='rgb_array')
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = HLTCN(obs_size, action_size)
    try:
        model.load_state_dict(torch.load('balancing_model.pth'))
        print("Successfully loaded 'balancing_model.pth'")
    except FileNotFoundError:
        print("Model file 'balancing_model.pth' not found. Please run train_balancing.py first.")
        return
    model.eval()
    
    # Get layer sizes for the drawing function
    h_low_size, h_high_size = model.h_low_size, model.h_high_size

    obs, _ = env.reset()
    h_low, h_high = model.init_hidden(batch_size=1)
    
    running = True
    print("Running live visualization... Close the window to stop.")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Simulation Step ---
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            action, _, _, h_low_next, h_high_next = model(obs_tensor, h_low, h_high)
        
        h_low, h_high = h_low_next, h_high_next
        obs, _, done, _, _ = env.step(action.item())
        
        if done:
            obs, _ = env.reset()
            h_low, h_high = model.init_hidden(batch_size=1)

        # --- Rendering ---
        # Render environment to an array
        sim_frame = env.render()
        sim_frame_surface = pygame.surfarray.make_surface(sim_frame.transpose(1, 0, 2))
        
        # Scale the simulation frame to fit the panel
        sim_surface.blit(pygame.transform.scale(sim_frame_surface, (SIM_PANEL_WIDTH, SCREEN_HEIGHT)), (0, 0))
        
        # Draw the network activations
        draw_network(net_surface, h_low, h_high, obs, h_low_size, h_high_size, obs_size)
        
        # Blit panels to the main screen
        screen.blit(sim_surface, (0, 0))
        screen.blit(net_surface, (SIM_PANEL_WIDTH, 0))
        
        pygame.display.flip()
        clock.tick(50) # Limit to 50 FPS

    env.close()
    pygame.quit()

if __name__ == "__main__":
    run_visualization()

