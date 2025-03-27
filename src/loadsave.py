import pickle
import numpy as np

def save_neural_network(model, filename):
    """
    Save a neural network model to a file.
    
    Parameters:
    -----------
    model : NeuralNetwork
        The neural network model to save.
    filename : str
        Path to the file where the model will be saved.
    """
    # Create a dictionary with all the model's important attributes
    model_state = {
        'layer_sizes': model.layer_sizes,
        'hidden_layer_activation_functions': model.hidden_layer_activation_functions,
        'output_layer_activation_function': model.output_layer_activation_function,
        'loss_function': model.loss_function,
        'learning_rate': model.learning_rate,
        'max_iter': model.max_iter,
        'batch_size': model.batch_size,
        'initialization_method': model.initialization_method,
        'verbose': model.verbose,
        'seed': model.seed,
        
        # Regularization parameters
        'l1_lambda': model.l1_lambda,
        'l2_lambda': model.l2_lambda,
        
        # Optimizer parameters
        'optimizer': model.optimizer,
        'beta1': model.beta1,
        'beta2': model.beta2,
        'epsilon': model.epsilon,
        
        # Model parameters
        'weights': model.weights,
        'biases': model.biases,
        'losses': model.losses,
        'gradients': model.gradients,
        'val_losses': model.val_losses,
        
        # Adam optimizer parameters if applicable
        'm_weights': model.m_weights if hasattr(model, 'm_weights') else None,
        'v_weights': model.v_weights if hasattr(model, 'v_weights') else None,
        'm_biases': model.m_biases if hasattr(model, 'm_biases') else None,
        'v_biases': model.v_biases if hasattr(model, 'v_biases') else None,
        't': model.t if hasattr(model, 't') else 0
    }
    
    # Use pickle to serialize the model state
    with open(filename, 'wb') as f:
        pickle.dump(model_state, f)
    
    print(f"Model saved to {filename}")

def load_neural_network(filename, neural_network_class):
    """
    Load a previously saved neural network model.
    
    Parameters:
    -----------
    filename : str
        Path to the file containing the saved model.
    neural_network_class : class
        The neural network class to use for reconstruction.
    
    Returns:
    --------
    NeuralNetwork
        A loaded neural network model with saved parameters.
    """
    # Load the model state
    with open(filename, 'rb') as f:
        model_state = pickle.load(f)
    
    # Create a new instance of the neural network with the saved parameters
    model = neural_network_class(
        layer_sizes=model_state['layer_sizes'],
        initialization_method=model_state['initialization_method'],
        hidden_layer_activation_functions=model_state['hidden_layer_activation_functions'][1:-1],  # Remove first and last None
        output_layer_activation_function=model_state['output_layer_activation_function'],
        loss_function=model_state['loss_function'],
        learning_rate=model_state['learning_rate'],
        max_iter=model_state['max_iter'],
        batch_size=model_state['batch_size'],
        optimizer=model_state['optimizer'],
        beta1=model_state['beta1'],
        beta2=model_state['beta2'],
        epsilon=model_state['epsilon'],
        l1_lambda=model_state['l1_lambda'],
        l2_lambda=model_state['l2_lambda'],
        verbose=model_state['verbose'],
        seed=model_state['seed']
    )
    
    # Restore model parameters
    model.weights = model_state['weights']
    model.biases = model_state['biases']
    model.losses = model_state['losses']
    model.gradient = model_state['gradients']
    model.val_losses = model_state['val_losses']
    
    # Restore Adam optimizer parameters if they exist
    if model_state['optimizer'] == 'adam':
        model.m_weights = model_state['m_weights']
        model.v_weights = model_state['v_weights']
        model.m_biases = model_state['m_biases']
        model.v_biases = model_state['v_biases']
        model.t = model_state['t']
    
    print(f"Model loaded from {filename}")
    
    return model