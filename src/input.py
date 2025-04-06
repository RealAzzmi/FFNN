from neural_network import relu, sigmoid, tanh, linear, softmax, elu, gelu
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
        'softmax': softmax,
        'elu': elu,
        'gel': gelu,
    }
    return activations.get(name, relu)  

# Function to map loss function names to actual functions
def get_loss_function(name):
    losses = {
        'binary_cross_entropy': binary_cross_entropy,
        'mean_squared_error': mean_squared_error,
        'categorical_cross_entropy': categorical_cross_entropy
    }
    return losses.get(name, binary_cross_entropy)  

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
                elif key == 'OPTIMIZER':
                    config['optimizer'] = value
                elif key == 'L1_LAMBDA':
                    config['l1_lambda'] = float(value)
                elif key == 'L2_LAMBDA':
                    config['l2_lambda'] = float(value)
                elif key == 'INITIALIZATION_METHOD':
                    config['initialization_method'] = value
                elif key == 'VERBOSE':
                    config['verbose'] = value.lower() == 'true'
    
    return config

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom neural network or scikit-learn MLPClassifier.")
    
    parser.add_argument('--default', action='store_true', help="Use default configuration from config.py")
    
    parser.add_argument('--layer_sizes', type=int, nargs='+', default=LAYER_SIZES, help="List of layer sizes (e.g., 784 128 64 10)")
    parser.add_argument('--hidden_activations', type=str, nargs='+', default=HIDDEN_LAYER_ACTIVATIONS, help="List of hidden layer activation functions (e.g., relu relu)")
    parser.add_argument('--output_activation', type=str, default=OUTPUT_LAYER_ACTIVATION, help="Output layer activation function (e.g., sigmoid)")
    parser.add_argument('--loss_function', type=str, default=LOSS_FUNCTION, help="Loss function (e.g., binary_cross_entropy)")
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help="Learning rate (e.g., 0.001)")
    parser.add_argument('--max_iter', type=int, default=MAX_ITER, help="Maximum number of iterations (e.g., 400)")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size (e.g., 64)")
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help="Optimizer to use (e.g., adam, sgd)")
    parser.add_argument('--l1_lambda', type=float, default=L1_LAMBDA, help="L1 regularization parameter (e.g., 0.01)")
    parser.add_argument('--l2_lambda', type=float, default=L2_LAMBDA, help="L2 regularization parameter (e.g., 0.01)")
    parser.add_argument('--initialization_method', type=str, default=INITIALIZATION_METHOD, help="Weight initialization method (e.g., he, xavier)")
    parser.add_argument('--verbose', type=bool, default=VERBOSE, help="Verbose mode (True/False)")

    parser.add_argument('--config_file', type=str, help="Path to configuration file")
    
    return parser.parse_args()

# Function to get configuration interactively
def get_config_interactively():

    activation_options = {
        '1': ('Linear', linear),
        '2': ('ReLU', relu),
        '3': ('Sigmoid', sigmoid),
        '4': ('Tanh', tanh),
        '5': ('Softmax', softmax),
        '6': ('ELU', elu),
        '7': ('GELU', gelu)
    }

    loss_function_options = {
        '1': ('Binary Cross-Entropy', binary_cross_entropy),
        '2': ('Mean Squared Error', mean_squared_error),
        '3': ('Categorical Cross-Entropy', categorical_cross_entropy)
    }

    def get_choice(prompt, options, default_key):
        print(prompt)
        for key, (name, _) in options.items():
            print(f"{key}. {name}")
        choice = input(f"Enter your choice (default: {default_key}): ").strip()
        return options.get(choice, options[default_key])[1]

    # Get user input for configuration
    layer_sizes = parse_layer_sizes(get_input(f"How many neurons in each layer? (default: {LAYER_SIZES}): ", str(LAYER_SIZES)))

    # Get hidden layer activation functions
    hidden_activations = []
    for i in range(len(layer_sizes) - 2):
        print(f"\nHidden Layer {i + 1}:")
        activation = get_choice(
            "Activation function you can choose are:",
            activation_options,
            default_key='2' 
        )
        hidden_activations.append(activation)

    # Get output layer activation function
    print("\nOutput Layer:")
    output_activation = get_choice(
        "Activation function you can choose are:",
        activation_options,
        default_key='3' 
    )

    # Get loss function
    print("\nLoss Function:")
    loss_function = get_choice(
        "Loss function you can choose are:",
        loss_function_options,
        default_key='1'
    )

    # Get other parameters
    learning_rate = float(get_input(f"Learning rate? (default: {LEARNING_RATE}): ", str(LEARNING_RATE)))
    max_iter = int(get_input(f"Maximum iterations? (default: {MAX_ITER}): ", str(MAX_ITER)))
    batch_size = int(get_input(f"Batch size? (default: {BATCH_SIZE}): ", str(BATCH_SIZE)))
    optimizer = get_input(f"Choose optimizer (adam/sgd, default: {OPTIMIZER}): ", str(OPTIMIZER))
    l1_lambda = float(get_input(f"L1 regularization parameter? (default: {L1_LAMBDA}): ", str(L1_LAMBDA)))
    l2_lambda = float(get_input(f"L2 regularization parameter? (default: {L2_LAMBDA}): ", str(L2_LAMBDA)))
    initialization_method = get_input(f"Choose weight initialization method (he/xavier, default: {INITIALIZATION_METHOD}): ", str(INITIALIZATION_METHOD))

    # Get verbose mode
    verbose = get_input("Would you like to see progress during training? (yes/no, default: yes): ", "yes").lower() == "yes"

    print("==================================================")
    print("Done getting neural network configuration")
    print("==================================================")

    return {
        'layer_sizes': layer_sizes,
        'hidden_activations': hidden_activations,
        'output_activation': output_activation,
        'loss_function': loss_function,
        'learning_rate': learning_rate,
        'max_iter': max_iter,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'l1_lambda': l1_lambda,
        'l2_lambda': l2_lambda,
        'initialization_method': initialization_method,
        'verbose': verbose
    }
