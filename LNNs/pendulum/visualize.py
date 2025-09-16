import torch
import yaml
import numpy as np
import networkx as nx
import json
import os
from tqdm import tqdm
import webbrowser
import http.server
import socketserver
import threading
import time

from environment import InvertedPendulum
from train import ActorCritic

def export_visualization_data():
    """
    Exports simulation data for the CfC network to a JSON file.
    """
    print("Starting visualization data export for CfC network...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    visuals_dir = 'visuals'
    os.makedirs(visuals_dir, exist_ok=True)
    
    log_path = config['training']['log_save_path']
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
        
        print(f"Generating training metrics plot and saving to '{visuals_dir}/'...")
        losses = logs['losses']
        rewards = logs['rewards']
            
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:red'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(losses, color=color, alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)
            
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Total Reward', color=color)

        if len(rewards) > 50:
            reward_smooth = np.convolve(rewards, np.ones(50)/50, mode='valid')
            ax2.plot(range(49, len(rewards)), reward_smooth, color=color)
        else:
            ax2.plot(rewards, color=color)

        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.title('Training Metrics')
        plt.savefig(f'{visuals_dir}/training_metrics.png')
        plt.close()
            
        print("Plot generated.")

    except (FileNotFoundError, ImportError) as e:
        if isinstance(e, FileNotFoundError):
            print(f"Warning: Training log file not found at '{log_path}'. Skipping plot generation.")
        else:
            print("Warning: Matplotlib not found. `pip install matplotlib`")


    env = InvertedPendulum(config)
    model = ActorCritic(config)
    
    model_path = config['training']['model_save_path']
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded final model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        return

    model.eval()

    print("Generating 3-layer network layout...")
    G = nx.Graph()
    input_size = config['environment']['state_size']
    hidden_size = config['network']['hidden_size']
    output_size = config['environment']['action_size']
    
    input_nodes = [f"in_{i}" for i in range(input_size)]
    hidden_nodes = [f"h_{i}" for i in range(hidden_size)]
    output_nodes = [f"out_{i}" for i in range(output_size)]
    
    G.add_nodes_from(input_nodes, layer=0)
    G.add_nodes_from(hidden_nodes, layer=1)
    G.add_nodes_from(output_nodes, layer=2)
    
    network_pos = nx.multipartite_layout(G, subset_key='layer', align='vertical')
    serializable_pos = {str(k): v.tolist() for k, v in network_pos.items()}

    print("Running simulation to collect animation frames...")
    state, _ = env.reset()
    hidden_state = None
    frames_data = []
    
    theta_thresh = config['environment']['theta_threshold_radians']
    x_vis_thresh = 2.4 
    
    for _ in tqdm(range(config['visualization']['max_frames']), desc="Exporting Frames"):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_mean, _, _, new_hidden_state = model(state_tensor, hidden_state)
        
        hidden_state = new_hidden_state
        action_np = action_mean.detach().numpy().flatten()
        
        next_state, _, done, _, _ = env.step(action_np)
        state = next_state

        norm_activations = (np.clip(hidden_state.squeeze().numpy(), -1, 1) + 1) / 2.0
        
        norm_x = (np.clip(state[0], -x_vis_thresh, x_vis_thresh) + x_vis_thresh) / (2 * x_vis_thresh)
        norm_x_dot = (np.clip(state[1], -3, 3) + 3) / 6.0
        norm_theta = (np.clip(state[2], -theta_thresh, theta_thresh) + theta_thresh) / (2 * theta_thresh)
        norm_theta_dot = (np.clip(state[3], -4, 4) + 4) / 8.0
        norm_action = (action_np[0] + 1) / 2.0

        frame = {
            "cart_x": float(state[0]),
            "pole_theta": float(state[2]),
            "activations": norm_activations.tolist(),
            "inputs": [float(norm_x), float(norm_x_dot), float(norm_theta), float(norm_theta_dot)],
            "output": float(norm_action)
        }
        frames_data.append(frame)

        if done:
            state, _ = env.reset()
            hidden_state = None

    output_data = {
        "network_layout": serializable_pos,
        "frames": frames_data,
        "config": { "x_threshold": env.x_threshold, "pole_length": env.length }
    }

    save_path = "visualization_data.json"
    with open(save_path, 'w') as f:
        json.dump(output_data, f)
        
    print(f"\n✅ Success! Data exported to '{save_path}'.")

def start_server(port=8000):
    handler = http.server.SimpleHTTPRequestHandler
    while True:
        try:
            httpd = socketserver.TCPServer(("", port), handler)
            print(f"\n🌐 Serving at http://localhost:{port}")
            print("   The visualization should open automatically in your browser.")
            print("   Press Ctrl+C in this terminal to stop the server.")
            httpd.serve_forever()
        except OSError:
            print(f"Port {port} is already in use, trying next one...")
            port += 1
        except KeyboardInterrupt:
            break
    return port

if __name__ == '__main__':
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    
    export_visualization_data()

    PORT = 8000
    server_thread = threading.Thread(target=start_server, args=(PORT,))
    server_thread.daemon = True 
    server_thread.start()
    time.sleep(1) 
    url = f"http://localhost:{PORT}/visualization.html"
    webbrowser.open_new_tab(url)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server and exiting.")