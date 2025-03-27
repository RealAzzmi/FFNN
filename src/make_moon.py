import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

from neural_network import NeuralNetwork
from visualizer import plot_decision_boundary, plot_confusion_matrices, plot_weight_distribution, plot_weight_gradient_distribution, plot_loss_curve, plot_decision_boundaries
from neural_network import relu, sigmoid, tanh, linear, softmax
from neural_network import mean_squared_error, categorical_cross_entropy, binary_cross_entropy
from input import parse_args, parse_config_file, get_config_interactively
from config import *

def use_make_moon(nn):
    print("Generating dataset...")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print("\nTraining custom neural network...")
    custom_start_time = time.time()
    train_losses, final_gradients = nn.fit(X_train, y_train)
    custom_training_time = time.time() - custom_start_time

    # Evaluate Neural Network model
    y_pred_proba = nn.predict(X_test)
    y_pred_nn = (y_pred_proba > 0.5).astype(int)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    conf_matrix_custom = confusion_matrix(y_test, y_pred_nn)

    # Train scikit-learn MLP
    print("\nTraining scikit-learn MLPClassifier...")
    sklearn_nn = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu', 
        solver='sgd',
        learning_rate_init=0.001,
        max_iter=400,
        batch_size=64
    )
    sklearn_start_time = time.time()
    sklearn_nn.fit(X_train, y_train)
    sklearn_training_time = time.time() - sklearn_start_time

    # Evaluate scikit-learn model
    y_pred_sklearn = sklearn_nn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)

    return nn, train_losses, final_gradients, custom_training_time, accuracy_nn, conf_matrix_custom, sklearn_training_time, accuracy_sklearn, conf_matrix_sklearn


def make_moon():
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
    print("Generating dataset...")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    

    # Initialize and train the neural network
    print("\nTraining custom neural network...")
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
    custom_start_time = time.time()
    train_losses, final_gradients = nn.fit(X_train, y_train)
    custom_training_time = time.time() - custom_start_time

    # Evaluate Neural Network model
    y_pred_proba = nn.predict(X_test)
    y_pred_nn = (y_pred_proba > 0.5).astype(int)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    conf_matrix_custom = confusion_matrix(y_test, y_pred_nn)
    print(f"Neural Network Accuracy: {accuracy_nn:.4f}")

    # Train scikit-learn MLP
    print("\nTraining scikit-learn MLPClassifier...")
    sklearn_nn = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu', 
        solver='sgd',
        learning_rate_init=0.001,
        max_iter=400,
        batch_size=64
    )
    sklearn_start_time = time.time()
    sklearn_nn.fit(X_train, y_train)
    sklearn_training_time = time.time() - sklearn_start_time

    # Evaluate scikit-learn model
    y_pred_sklearn = sklearn_nn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)
    print(f"scikit-learn MLPClassifier Accuracy: {accuracy_sklearn:.4f}")

    print("\n----- Results -----")
    print(f"Custom Neural Network Accuracy: {accuracy_nn:.4f}")
    print(f"scikit-learn MLPClassifier Accuracy: {accuracy_sklearn:.4f}")
    print(f"Custom Neural Network Training Time: {custom_training_time:.2f} seconds")
    print(f"scikit-learn MLPClassifier Training Time: {sklearn_training_time:.2f} seconds")

    is_plot_weight_distribution = input("Do you wish to show weight distribution? (y/n)")
    if is_plot_weight_distribution=="y":
        plot_weight_distribution(nn.weights)
        plot_weight_gradient_distribution(final_gradients)
    
    is_plot_confusion_matrices = input("Do you wish to show confusion matrices? (y/n)")
    if is_plot_confusion_matrices=="y":
        plot_confusion_matrices(conf_matrix_custom, conf_matrix_sklearn)

    is_plot_loss_curve = input("Do you wish to show training history loss curve? (y/n)")
    if is_plot_loss_curve=="y":
        plot_loss_curve(train_losses)
    
    is_plot_decision_boundaries = input("Do you wish to show decision boundaries? (y/n)")
    if is_plot_decision_boundaries=="y":
        plot_decision_boundaries(
            X_test, y_test, X_test, y_test,
            [nn, sklearn_nn],
            ["Custom Neural Network", "scikit-learn MLPClassifier"]
        )


if __name__ == "__main__":
    make_moon()