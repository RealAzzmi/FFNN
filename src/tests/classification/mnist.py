import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from loadsave import save_neural_network, load_neural_network

# Import your neural network implementation
# Assuming it's in a file called neural_network.py in the same directory
from neural_network import NeuralNetwork, relu, softmax, categorical_cross_entropy

def load_mnist_data(n_samples=None):
    """Load MNIST dataset and preprocess it."""
    print("Loading MNIST dataset...")
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    
    # Convert labels to integers
    y = y.astype(int)
    
    # Limit the number of samples if specified
    if n_samples is not None:
        X = X[:n_samples]
        y = y[:n_samples]
    
    # Scale features to [0, 1] range
    X = X / 255.0
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_val.shape[0]} test samples")
    return X_train, X_val, y_train, y_val

def one_hot_encode(y, n_classes=10):
    """Convert integer labels to one-hot encoded vectors."""
    y_one_hot = np.zeros((y.shape[0], n_classes))
    for i, label in enumerate(y):
        y_one_hot[i, label] = 1
    return y_one_hot

def visualize_results(custom_losses, sklearn_accuracy, custom_accuracy):
    """Visualize the training loss and comparison of accuracies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(custom_losses)
    ax1.set_title('Custom Neural Network Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracy comparison
    models = ['Scikit-Learn MLP', 'Custom Neural Network']
    accuracies = [sklearn_accuracy, custom_accuracy]
    
    ax2.bar(models, accuracies, color=['blue', 'orange'])
    ax2.set_title('Model Accuracy Comparison')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    ax2.grid(axis='y')
    
    for i, v in enumerate(accuracies):
        ax2.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    # plt.savefig('mnist_comparison_results.png')
    plt.show()

def visualize_confusion_matrices(y_val, sklearn_preds, custom_preds):
    """Visualize confusion matrices for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sklearn MLP confusion matrix
    cm_sklearn = confusion_matrix(y_val, sklearn_preds)
    im1 = ax1.imshow(cm_sklearn, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Scikit-Learn MLP Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_xticks(np.arange(10))
    ax1.set_yticks(np.arange(10))
    fig.colorbar(im1, ax=ax1)

    # Display values inside confusion matrix
    for i in range(cm_sklearn.shape[0]):
        for j in range(cm_sklearn.shape[1]):
            ax1.text(j, i, str(cm_sklearn[i, j]), ha='center', va='center', color='white' if cm_sklearn[i, j] > cm_sklearn.max() / 2 else 'black')
    
    # Custom NN confusion matrix
    cm_custom = confusion_matrix(y_val, custom_preds)
    im2 = ax2.imshow(cm_custom, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('Custom Neural Network Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_xticks(np.arange(10))
    ax2.set_yticks(np.arange(10))
    fig.colorbar(im2, ax=ax2)

    # Display values inside confusion matrix
    for i in range(cm_custom.shape[0]):
        for j in range(cm_custom.shape[1]):
            ax2.text(j, i, str(cm_custom[i, j]), ha='center', va='center', color='white' if cm_custom[i, j] > cm_custom.max() / 2 else 'black')
    
    plt.tight_layout()
    # plt.savefig('mnist_confusion_matrices.png')
    plt.show()

def use_mnist(custom_nn):
    X_train, X_val, y_train, y_val = load_mnist_data(n_samples=1000)
    y_train_one_hot = one_hot_encode(y_train)
    y_val_one_hot = one_hot_encode(y_val)

    #Train Sklearn
    print("\nTraining Scikit-Learn MLP Classifier...")
    start_time = time.time()
    
    sklearn_mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate_init=0.01,
        max_iter=50,
        verbose=True,
        random_state=42
    )
    
    sklearn_mlp.fit(X_train, y_train)
    sklearn_time = time.time() - start_time
    sklearn_preds = sklearn_mlp.predict(X_val)
    sklearn_accuracy = accuracy_score(y_val, sklearn_preds)
    conf_matrix_sklearn = confusion_matrix(y_val, sklearn_preds)

    # Train Custom Neural Network
    print("\nTraining Custom Neural Network...")
    start_time = time.time()
    custom_losses = []
    custom_losses, final_gradients = custom_nn.fit(X_train.values, y_train_one_hot, X_val.values, y_val_one_hot)
    custom_time = time.time() - start_time
    custom_probs = custom_nn.predict(X_val.values)
    custom_preds = np.argmax(custom_probs, axis=1)
    custom_accuracy = accuracy_score(y_val, custom_preds)
    conf_matrix_custom = confusion_matrix(y_val, custom_preds)

    return custom_nn, custom_losses, final_gradients, custom_time, custom_accuracy, conf_matrix_custom, sklearn_time, sklearn_accuracy, conf_matrix_sklearn, X_val, y_val, sklearn_mlp



def main():
    # Load and preprocess the MNIST dataset
    # Use a smaller subset for faster training
    X_train, X_val, y_train, y_val = load_mnist_data(n_samples=1000)
    
    # One-hot encode the target for custom neural network
    y_train_one_hot = one_hot_encode(y_train)
    y_val_one_hot = one_hot_encode(y_val)
    
    # Train Scikit-Learn MLP Classifier
    print("\nTraining Scikit-Learn MLP Classifier...")
    start_time = time.time()
    
    sklearn_mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate_init=0.01,
        max_iter=50,
        verbose=True,
        random_state=42
    )
    
    sklearn_mlp.fit(X_train, y_train)
    sklearn_time = time.time() - start_time
    print(f"Scikit-Learn MLP training time: {sklearn_time:.2f} seconds")

    is_pretrained_model = input("Do you wish to use pretrained model? (y/n)")
    if is_pretrained_model == "y":
        pretrained_model_file = input("Pretrained model file name: ")
        custom_nn = load_neural_network(pretrained_model_file, NeuralNetwork)
    else:  
        # Input size is 784 (28x28 pixels), output size is 10 (digit classes 0-9)
        custom_nn = NeuralNetwork(
            layer_sizes=[784, 256, 128, 10],
            hidden_layer_activation_functions=[relu, relu],
            output_layer_activation_function=softmax,
            loss_function=categorical_cross_entropy,
            learning_rate=0.01,
            max_iter=50,
            batch_size=256,
            optimizer="adam",
            initialization_method="he",
            verbose=True
        )
        # Train Custom Neural Network
        print("\nTraining Custom Neural Network...")
    
    start_time = time.time()
    
    # Prepare the data for custom NN
    custom_losses = []
    
    # Train the custom neural network one sample at a time
    custom_losses, _ = custom_nn.fit(X_train.values, y_train_one_hot, X_val.values, y_val_one_hot)

    
    
    custom_time = time.time() - start_time
    print(f"Custom Neural Network training time: {custom_time:.2f} seconds")
    
    # Evaluate Scikit-Learn MLP
    sklearn_preds = sklearn_mlp.predict(X_val)
    sklearn_accuracy = accuracy_score(y_val, sklearn_preds)
    print(f"\nScikit-Learn MLP Accuracy: {sklearn_accuracy:.4f}")
    print("\nScikit-Learn MLP Classification Report:")
    print(classification_report(y_val, sklearn_preds))
    
    # Evaluate Custom Neural Network
    custom_probs = custom_nn.predict(X_val.values)
    custom_preds = np.argmax(custom_probs, axis=1)
    custom_accuracy = accuracy_score(y_val, custom_preds)
    print(f"\nCustom Neural Network Accuracy: {custom_accuracy:.4f}")
    print("\nCustom Neural Network Classification Report:")
    print(classification_report(y_val, custom_preds))

    #Save results
    save_neural_network(custom_nn, 'mnist_neural_network.pkl')

    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(custom_losses, sklearn_accuracy, custom_accuracy)
    visualize_confusion_matrices(y_val, sklearn_preds, custom_preds)
    
    # Compare training time
    print(f"\nTraining time comparison:")
    print(f"Scikit-Learn MLP: {sklearn_time:.2f} seconds")
    print(f"Custom Neural Network: {custom_time:.2f} seconds")
    print(f"Ratio (Custom/Scikit-Learn): {custom_time/sklearn_time:.2f}x")

if __name__ == "__main__":
    main()