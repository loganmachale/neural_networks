import networkx as nx

def create_network_graph(model):
    """
    Creates a NetworkX graph object from the model's structure.
    
    Args:
        model (LiquidNeuralNetwork): The network model instance.
    
    Returns:
        nx.Graph: A NetworkX graph representing the network.
    """
    G = nx.Graph()
    
    # Add neuron nodes
    for i in range(model.num_neurons):
        G.add_node(f"N{i}", type='neuron')

    # Add edges from source neurons to target neurons based on adjacency matrix
    adj_matrix = model.neuron_adj.detach().numpy()
    rows, cols = adj_matrix.nonzero()
    for source_neuron, target_neuron in zip(rows, cols):
        G.add_edge(f"N{source_neuron}", f"N{target_neuron}")
        
    return G