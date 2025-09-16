Biologically Inspired Liquid Neural Network for Inverted Pendulum
This project implements and trains a biologically inspired Liquid Time-Constant Neural Network (LTC) to solve the classic control problem of balancing an inverted pendulum. The network architecture is based on the principles outlined in the provided research paper, with the unique addition of dendrites that allow for signal interference.

Project Structure
README.md: This file, providing an overview of the project.

config.yaml: A configuration file to customize network parameters, training settings, and environment variables.

network.py: Defines the LiquidNeuralNetwork class, including the custom dendrite and neuron dynamics based on Ordinary Differential Equations (ODEs).

environment.py: Implements the InvertedPendulum environment, handling the physics simulation and reward calculation.

train.py: The main script for training the neural network. It orchestrates the interaction between the network and the environment, manages the training loop, and saves the trained model.

visualize.py: A script for visualizing the trained network's performance, its structure, and training metrics.

utils.py: Contains utility functions, primarily for aiding in the visualization of the network graph.

requirements.txt: A list of Python dependencies required to run the project.

Getting Started
Prerequisites
Python 3.8 or newer

Pip package manager

Installation
Clone the repository:

git clone <repository-url>
cd <repository-directory>

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the dependencies:

pip install -r requirements.txt

Configuration
Before running the training or visualization, you can adjust the parameters in the config.yaml file. This includes:

Biological Design Protocol (BDP): Number of neurons, dendrite connectivity, etc.

Network Parameters: Time constants, solver steps.

Environment Parameters: Physics constants for the pendulum.

Training Parameters: Learning rate, number of episodes, reward/punishment settings.

Training the Network
To start the training process, run the train.py script from your terminal:

python train.py

The script will periodically save the trained model (trained_model.pth) and log training data (training_logs.json) to the project directory.

Visualizing the Results
After training is complete, you can visualize the results by running the visualize.py script:

python visualize.py

This will:

Display a window animating the trained model balancing the inverted pendulum.

Generate and save a visualization of the network's structure as network_structure.png.

Display plots of the training loss and task performance over episodes.

How It Works
The core of this project is the LiquidNeuralNetwork which is a type of Ordinary Differential Equation (ODE) based recurrent neural network.

Dendrites: A unique feature of this implementation is the concept of dendrites. Each neuron receives inputs exclusively through its dendrites. Multiple other neurons can connect to a single dendrite, where their signals interact. This interaction is modulated by a trainable parameter, allowing for either constructive or destructive interference.

ODE Solver: The network's state is continuous and evolves over time according to a system of differential equations. We use a numerical ODE solver (Euler's method in this implementation) to approximate the network's state at discrete time steps.

Reinforcement Learning: The network is trained using a simple reinforcement learning algorithm. It receives the state of the pendulum and its pivot as input and outputs a force to apply to the pivot. It is rewarded for keeping the pendulum upright and penalized for letting it fall.