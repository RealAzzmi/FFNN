from neural_network import relu, sigmoid, tanh, linear, softmax
from neural_network import mean_squared_error, categorical_cross_entropy, binary_cross_entropy
from config import *
import argparse

# Function to map activation function names to actual functions
def get_activation_function(name):
    activations = {
        'relu': relu,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'linear': linear,
        'softmax': softmax
    }
    return activations.get(name, relu)  # Default to relu if not found

# Function to map loss function names to actual functions
def get_loss_function(name):
    losses = {
        'binary_cross_entropy': binary_cross_entropy,
        'mean_squared_error': mean_squared_error,
        'categorical_cross_entropy': categorical_cross_entropy
    }
    return losses.get(name, binary_cross_entropy)  # Default to binary_cross_entropy if not found

# Function to get user input with default value
def get_input(prompt, default):
    user_input = input(prompt)
    return default if user_input.strip() == "" else user_input

# Function to parse layer sizes from user input
def parse_layer_sizes(input_str):
    try:
        return list(map(int, input_str.strip().split()))
    except ValueError:
        print("Invalid input for layer sizes. Using default value.")
        return LAYER_SIZES

# Function to parse hidden activations from user input
def parse_hidden_activations(input_str):
    try:
        return [get_activation_function(act) for act in input_str.strip().split()]
    except ValueError:
        print("Invalid input for hidden activations. Using default value.")
        return HIDDEN_LAYER_ACTIVATIONS

# Function to parse a configuration file
def parse_config_file(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'LAYER_SIZES':
                    config['layer_sizes'] = list(map(int, value.strip('[]').split(',')))
                elif key == 'HIDDEN_LAYER_ACTIVATIONS':
                    config['hidden_activations'] = [get_activation_function(act.strip()) for act in value.strip('[]').split(',')]
                elif key == 'OUTPUT_LAYER_ACTIVATION':
                    config['output_activation'] = get_activation_function(value)
                elif key == 'LOSS_FUNCTION':
                    config['loss_function'] = get_loss_function(value)
                elif key == 'LEARNING_RATE':
                    config['learning_rate'] = float(value)
                elif key == 'MAX_ITER':
                    config['max_iter'] = int(value)
                elif key == 'BATCH_SIZE':
                    config['batch_size'] = int(value)
                elif key == 'VERBOSE':
                    config['verbose'] = value.lower() == 'true'
    
    return config

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom neural network or scikit-learn MLPClassifier.")
    
    parser.add_argument('--default', action='store_true', help="Use default configuration from config.py")
    
    parser.add_argument('--layer_sizes', type=int, nargs='+', default=LAYER_SIZES, help="List of layer sizes (e.g., 2 32 16 1)")
    parser.add_argument('--hidden_activations', type=str, nargs='+', default=['relu', 'relu'], help="List of hidden layer activation functions (e.g., relu relu)")
    parser.add_argument('--output_activation', type=str, default='sigmoid', help="Output layer activation function (e.g., sigmoid)")
    parser.add_argument('--loss_function', type=str, default='binary_cross_entropy', help="Loss function (e.g., binary_cross_entropy)")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (e.g., 0.001)")
    parser.add_argument('--max_iter', type=int, default=400, help="Maximum number of iterations (e.g., 400)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (e.g., 64)")
    parser.add_argument('--verbose', type=bool, default=True, help="Verbose mode (True/False)")

    parser.add_argument('--config_file', type=str, help="Path to configuration file")
    
    return parser.parse_args()

# Function to get configuration interactively
def get_config_interactively():
    banner = r"""
    ███████╗███████╗███╗   ██╗███╗   ██╗
    ██╔════╝██╔════╝████╗  ██║████╗  ██║
    █████╗  █████╗  ██╔██╗ ██║██╔██╗ ██║
    ██╔══╝  ██╔══╝  ██║╚██╗██║██║╚██╗██║
    ██║     ██║     ██║ ╚████║██║ ╚████║
    ╚═╝     ╚═╝     ╚═╝  ╚═══╝╚═╝  ╚═══╝
    """
    print("\033[96m" + "=" * 50)
    print(banner)
    print("=" * 50)
    print("\033[92mWelcome to \033[1mFFNN\033[0m\033[92m by Azmi, Shulha, and Nabila.\033[0m")
    print("\033[93mYou can start by configuring the FFNN yourself \nor go with the default settings.\033[0m")
    print("\033[96m" + "=" * 50 + "\033[0m")

    layer_sizes = parse_layer_sizes(get_input(f"How many neurons in each layer? (default: {LAYER_SIZES}): ", str(LAYER_SIZES)))
    hidden_activations = parse_hidden_activations(get_input(f"Hidden layer activation functions? (default: [relu, relu]): ", str(HIDDEN_LAYER_ACTIVATIONS)))
    output_activation = get_activation_function(get_input(f"Output layer activation function? (default: {OUTPUT_LAYER_ACTIVATION.__name__}): ", OUTPUT_LAYER_ACTIVATION.__name__))
    loss_function = get_loss_function(get_input(f"Loss function? (default: {LOSS_FUNCTION.__name__}): ", LOSS_FUNCTION.__name__))
    learning_rate = float(get_input(f"Learning rate? (default: {LEARNING_RATE}): ", str(LEARNING_RATE)))
    max_iter = int(get_input(f"Maximum iterations? (default: {MAX_ITER}): ", str(MAX_ITER)))
    batch_size = int(get_input(f"Batch size? (default: {BATCH_SIZE}): ", str(BATCH_SIZE)))
    verbose = get_input(f"Verbose mode? (True/False, default: {VERBOSE}): ", str(VERBOSE)).lower() == "true"

    print("==================================================")
    print("Now, please wait for your program to run.")
    print("==================================================")

    return {
        'layer_sizes': layer_sizes,
        'hidden_activations': hidden_activations,
        'output_activation': output_activation,
        'loss_function': loss_function,
        'learning_rate': learning_rate,
        'max_iter': max_iter,
        'batch_size': batch_size,
        'verbose': verbose
    }
