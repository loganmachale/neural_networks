import torch
import numpy as np
import pygame
import time

from environment import ScheduledCartPoleEnv
from model import HLTCN

# --- Visualization Parameters ---
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 600
SIM_PANEL_WIDTH = 900
NET_PANEL_WIDTH = 600
BACKGROUND_COLOR = (10, 10, 50)
NEURON_RADIUS = 8
LAYER_SPACING = 300

def draw_network(surface, h_low, h_high, obs, low_size, high_size):
    """Draws the neural network activation on the given Pygame surface."""
    surface.fill(BACKGROUND_COLOR)
    font = pygame.font.Font(None, 24)

    # Handle the initial None state after an episode reset
    if h_low is None:
        h_low = torch.zeros(1, low_size)
    if h_high is None:
        h_high = torch.zeros(1, high_size)

    # Normalize activations to be in a visible range (e.g., using tanh)
    h_low_norm = torch.tanh(h_low).squeeze().numpy()
    h_high_norm = torch.tanh(h_high).squeeze().numpy()
    obs_norm = np.tanh(obs)

    # --- Draw Input Neurons (Observations) ---
    title = font.render("Inputs (Observation)", True, (200, 200, 200))
    surface.blit(title, (20, 20))
    for i, activation in enumerate(obs_norm):
        color_val = int((activation + 1) / 2 * 255)
        color = (color_val, color_val, color_val)
        pos_x = 50
        pos_y = 60 + i * (NEURON_RADIUS * 2 + 5)
        pygame.draw.circle(surface, color, (pos_x, pos_y), NEURON_RADIUS)

    # --- Draw High-Level (Slow) Neurons ---
    title = font.render("High-Level (Slow)", True, (200, 200, 200))
    surface.blit(title, (150, 20))
    for i, activation in enumerate(h_high_norm):
        color_val = int((activation + 1) / 2 * 255)
        color = (color_val, color_val, color_val)
        pos_x = 180
        pos_y = 60 + i * (NEURON_RADIUS * 2 + 5)
        pygame.draw.circle(surface, color, (pos_x, pos_y), NEURON_RADIUS)

    # --- Draw Low-Level (Fast) Neurons ---
    title = font.render("Low-Level (Fast)", True, (200, 200, 200))
    surface.blit(title, (280, 20))
    for i, activation in enumerate(h_low_norm):
        color_val = int((activation + 1) / 2 * 255)
        color = (color_val, color_val, color_val)
        pos_x = 310
        pos_y = 60 + i * (NEURON_RADIUS * 2 + 3)
        pygame.draw.circle(surface, color, (pos_x, pos_y), NEURON_RADIUS)

def run_visualization():
    # --- Initialization ---
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("HLTCN Visualization")
    sim_surface = pygame.Surface((SIM_PANEL_WIDTH, SCREEN_HEIGHT))
    net_surface = pygame.Surface((NET_PANEL_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    # --- Load Environment and Model ---
    # CRITICAL FIX: Change render_mode to 'rgb_array' to get frames as data
    env = ScheduledCartPoleEnv(render_mode='rgb_array')
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = HLTCN(obs_size, action_size)
    low_level_size = model.ltc_low.hidden_size
    high_level_size = model.ltc_high.hidden_size
    try:
        model.load_state_dict(torch.load('balancing_model.pth'))
    except FileNotFoundError:
        print("Model file 'balancing_model.pth' not found. Please run train.py first.")
        return
    model.eval()

    obs, _ = env.reset()
    h_low, h_high = None, None

    # --- Main Loop ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Model Inference ---
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            action, _, _, h_low_next, h_high_next = model(obs_tensor, h_low, h_high)

        h_low, h_high = h_low_next, h_high_next

        # --- Environment Step ---
        obs, _, done, _, _ = env.step(action.item())
        if done:
            print("Episode finished. Resetting.")
            obs, _ = env.reset()
            h_low, h_high = None, None
            time.sleep(1)

        # --- Drawing ---
        screen.fill((0, 0, 0))

        # 1. Render environment to an RGB array
        rgb_array = env.render()
        if rgb_array is not None:
            # Convert the numpy array to a Pygame surface
            frame_surface = pygame.surfarray.make_surface(rgb_array.transpose(1, 0, 2))
            # Scale the frame to fit our simulation panel and blit it
            sim_surface.blit(pygame.transform.scale(frame_surface, (SIM_PANEL_WIDTH, SCREEN_HEIGHT)), (0, 0))

        # 2. Draw network activations to its surface
        draw_network(net_surface, h_low, h_high, obs, low_level_size, high_level_size)

        # 3. Blit both panels to the main screen
        screen.blit(sim_surface, (0, 0))
        screen.blit(net_surface, (SIM_PANEL_WIDTH, 0))

        # 4. Update the display once at the end of the loop
        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    run_visualization()

