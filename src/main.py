from input import parse_args, parse_config_file, get_config_interactively
from config import *
from loadsave import load_neural_network, save_neural_network
from visualizer import plot_confusion_matrices, plot_loss_curve, plot_weight_distribution, plot_weight_gradient_distribution
from neural_network import NeuralNetwork
from make_moon import use_make_moon
from tests.classification.mnist import use_mnist

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
    print("\033[92mWelcome to \033[1mFFNN\033[0m\033[92m by Azmi, Shulha, and Nabila.\033[0m")
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
        verbose=config['verbose']
        )
    
    #Choose and Run Model on Dataset
    test_dataset_options = {
        '1': ('Make Moon'),
        '2': ('Mnist')
    }
    print("Test Dataset options you can choose are")
    for key, name in test_dataset_options.items():
        print(f"{key}. {name}")
    choice = input(f"Enter your choice (default: {1}): ").strip()
    if (choice == "1"):
        nn, train_losses, final_gradients, custom_training_time, accuracy_nn, conf_matrix_custom, sklearn_training_time, accuracy_sklearn, conf_matrix_sklearn = use_make_moon(custom_nn)
    elif (choice == "2"):
        nn, train_losses, final_gradients, custom_training_time, accuracy_nn, conf_matrix_custom, sklearn_training_time, accuracy_sklearn, conf_matrix_sklearn = use_mnist(custom_nn)
    #Print Result
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

    is_save = input("Do you wish to save the file? (y/n)")
    if is_save=="y":
        save_file_name = input("Input file name to save the trained model (ended with .pkl)")
        save_neural_network(nn, save_file_name)

    nn.visualize_neural_network()
    

if __name__ == "__main__":
    main()

