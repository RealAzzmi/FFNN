import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from neural_network import NeuralNetwork
from visualizer import plot_decision_boundaries
from neural_network import relu, sigmoid, tanh, linear, softmax
from neural_network import mean_squared_error, categorical_cross_entropy, binary_cross_entropy
from input import parse_args, parse_config_file, get_config_interactively
from config import *

def main():
    args = parse_args()

    if args.config_file:
        config = parse_config_file(args.config_file)
    elif args.default:
        config = {
            'layer_sizes': LAYER_SIZES,
            'hidden_activations': HIDDEN_LAYER_ACTIVATIONS,
            'output_activation': OUTPUT_LAYER_ACTIVATION,
            'loss_function': LOSS_FUNCTION,
            'learning_rate': LEARNING_RATE,
            'max_iter': MAX_ITER,
            'batch_size': BATCH_SIZE,
            'verbose': VERBOSE
        }
    else:
        config = get_config_interactively()

    # Sample Data
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Initialize and train the neural network
    nn = NeuralNetwork(
        layer_sizes=config['layer_sizes'],
        hidden_layer_activation_functions=config['hidden_activations'],
        output_layer_activation_function=config['output_activation'],
        loss_function=config['loss_function'],
        learning_rate=config['learning_rate'],
        max_iter=config['max_iter'],
        batch_size=config['batch_size'],
        verbose=config['verbose']
    )
    nn.fit(X_train, y_train)

    # Evaluate Neural Network model
    y_pred_nn = nn.predict(X_test)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    print(f"Neural Network Accuracy: {accuracy_nn:.4f}")

    # Train scikit-learn MLP
    sklearn_nn = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu', 
        solver='sgd',
        learning_rate_init=0.001,
        max_iter=400,
        batch_size=64
    )
    sklearn_nn.fit(X_train, y_train)

    # Evaluate scikit-learn model
    y_pred_sklearn = sklearn_nn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"scikit-learn MLPClassifier Accuracy: {accuracy_sklearn:.4f}")

    # Plot decision boundaries
    plot_decision_boundaries(
        X_test, y_test, X_test, y_test,
        [nn, sklearn_nn],
        ["Custom Neural Network", "scikit-learn MLPClassifier"]
    )

if __name__ == "__main__":
    main()