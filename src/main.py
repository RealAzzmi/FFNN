from input import parse_args, parse_config_file, get_config_interactively
from config import *
from loadsave import load_neural_network, save_neural_network
from visualizer import plot_confusion_matrices, plot_loss_curve, plot_weight_distribution, plot_weight_gradient_distribution
from neural_network import NeuralNetwork
from make_moon import use_make_moon
from tests.classification.mnist import use_mnist
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def main():
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
    print("\033[92mWelcome to \033[1mFFNN\033[0m\033[92m by Azmi, Nabila, and Shulha.\033[0m")
    print("\033[93mYou can start by configuring the FFNN yourself \nor go with the default settings or use pretrained model.\033[0m")
    print("\033[96m" + "=" * 50 + "\033[0m")

    is_pretrained_model = input("Do you wish to use pretrained model? (y/n)")
    if is_pretrained_model == "y":
        pretrained_model_file = input("Pretrained model file name: ")
        custom_nn = load_neural_network(pretrained_model_file, NeuralNetwork)

    else:  
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
                'optimizer': OPTIMIZER,
                'l1_lambda': L1_LAMBDA,
                'l2_lambda': L2_LAMBDA,
                'initialization_method': INITIALIZATION_METHOD,
                'verbose': VERBOSE
            }
        else:
            config = get_config_interactively()
        
        custom_nn = NeuralNetwork(
        layer_sizes=config['layer_sizes'],
        hidden_layer_activation_functions=config['hidden_activations'],
        output_layer_activation_function=config['output_activation'],
        loss_function=config['loss_function'],
        learning_rate=config['learning_rate'],
        max_iter=config['max_iter'],
        batch_size=config['batch_size'],
        optimizer=config['optimizer'],
        l1_lambda=config['l1_lambda'],
        l2_lambda=config['l2_lambda'],
        initialization_method=config['initialization_method'],
        verbose=config['verbose']
        )
    
    # Choose and Run Model on Dataset
    test_dataset_options = {
        '1': ('Mnist'),
        '2': ('Make Moon')
    }
    print("Test Dataset options you can choose are")
    for key, name in test_dataset_options.items():
        print(f"{key}. {name}")
    choice = input(f"Enter your choice (default: {1}): ").strip()
    if (choice == "1"):
        nn, train_losses, final_gradients, custom_training_time, accuracy_nn, conf_matrix_custom, sklearn_training_time, accuracy_sklearn, conf_matrix_sklearn, X_val, y_val, sklearn_mlp  = use_mnist(custom_nn)
    elif (choice == "2"):
        nn, train_losses, final_gradients, custom_training_time, accuracy_nn, conf_matrix_custom, sklearn_training_time, accuracy_sklearn, conf_matrix_sklearn, X_val, y_val, sklearn_mlp = use_make_moon(custom_nn)
    else:
        nn, train_losses, final_gradients, custom_training_time, accuracy_nn, conf_matrix_custom, sklearn_training_time, accuracy_sklearn, conf_matrix_sklearn, X_val, y_val, sklearn_mlp = use_mnist(custom_nn)


    # Evaluate Scikit-Learn MLP
    sklearn_preds = sklearn_mlp.predict(X_val)
    sklearn_accuracy = accuracy_score(y_val, sklearn_preds)
    print(f"\nScikit-Learn MLP Accuracy: {sklearn_accuracy:.4f}")
    print("\nScikit-Learn MLP Classification Report:")
    print(classification_report(y_val, sklearn_preds))
    
    # Evaluate Custom Neural Network
    custom_probs = custom_nn.predict(X_val)
    custom_preds = np.argmax(custom_probs, axis=1)
    custom_accuracy = accuracy_score(y_val, custom_preds)
    print(f"\nCustom Neural Network Accuracy: {custom_accuracy:.4f}")
    print("\nCustom Neural Network Classification Report:")
    print(classification_report(y_val, custom_preds))
    
    
    print("\n\033[94m===== FFNN Configuration Used =====\033[0m")
    print(f"Hidden Layer Activation Functions: {[fn.__name__ for fn in config['hidden_activations']]}")
    print(f"Output Layer Activation Function: {config['output_activation'].__name__}")
    print(f"Loss Function: {config['loss_function'].__name__}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Maximum Iterations: {config['max_iter']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"L1 Regularization Lambda: {config['l1_lambda']}")
    print(f"L2 Regularization Lambda: {config['l2_lambda']}")
    print(f"Weight Initialization Method: {config['initialization_method']}")
    print(f"Verbose Mode: {'Yes' if config['verbose'] else 'No'}")

    # Compare training time and accuracy
    print("+" + "-" * 66 + "+")
    print(f"|{' ':<20} | {'Custom':^20} | {'Scikit-Learn ':^20}|")
    print(f"|{' Metric':<20} | {'Neural Network':^20} | {'MLPClassifier ':^20}|")
    print("+" + "-" * 66 + "+")
    print(f"|{' Accuracy':<20} | {accuracy_nn:^20.4f} | {accuracy_sklearn:^20.4f}|")
    print(f"|{' Training Time (s)':<20} | {custom_training_time:^20.2f} | {sklearn_training_time:^20.2f}|")
    print("+" + "-" * 66 + "+")
    print(f"|{' Ratio (Custom/Scikit-Learn)':<43} | {f'{custom_training_time/sklearn_training_time:.2f}x':^20}|")
    print("+" + "-" * 66 + "+")


    # Weight and Gradient Distribution
    is_plot_weight_distribution = input("\nDo you wish to show weight distribution? (y/n)")
    if is_plot_weight_distribution=="y":
        plot_weight_distribution(nn.weights)
        plot_weight_gradient_distribution(final_gradients)
    
    # Confusion Matrices
    is_plot_confusion_matrices = input("\nDo you wish to show confusion matrices? (y/n)")
    if is_plot_confusion_matrices=="y":
        plot_confusion_matrices(conf_matrix_custom, conf_matrix_sklearn)

    is_plot_loss_curve = input("\nDo you wish to show training history loss curve? (y/n)")
    if is_plot_loss_curve=="y":
        if nn.val_losses:
            plot_loss_curve(train_losses, nn.val_losses)
        else:
            print("\nValidation loss is not available — skipping validation curve.")
            plot_loss_curve(train_losses, None)

    # Visualize Neural Network
    is_neural_network = input("\nDo you wish to show the neural network? (y/n)")
    if is_neural_network=="y":
        nn.visualize_neural_network()

    # Save Model
    is_save = input("\nDo you wish to save the file? (y/n)")
    if is_save=="y":
        save_file_name = input("\nInput file name to save the trained model (ended with .pkl)")
        save_neural_network(nn, save_file_name)

    # Print Exit Message
    print("\033[\n92mThank you for using FFNN. Goodbye!\033[0m")

if __name__ == "__main__":
    main()












