import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Import the custom neural network implementation
# Make sure the file with your implementation is in the same directory
from neural_network import (
    NeuralNetwork, 
    relu, sigmoid, softmax, 
    categorical_cross_entropy
)

# Helper function for visualization
def plot_confusion_matrices(cm_custom, cm_sklearn, class_names):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax1)
    ax1.set_title('Custom Neural Network Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax2)
    ax2.set_title('Scikit-learn MLPClassifier Confusion Matrix')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    # plt.savefig('confusion_matrices_comparison.png')
    plt.close()

def plot_sample_images(X, y, class_names, num_samples=5):
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        for j in range(len(class_names)):
            # Find index of class j
            idx = np.where(y == j)[0][i]
            plt.subplot(num_samples, len(class_names), i * len(class_names) + j + 1)
            plt.imshow(X[idx].reshape(28, 28), cmap='gray')
            plt.title(f"{class_names[j]}")
            plt.axis('off')
    plt.tight_layout()
    # plt.savefig('fashion_mnist_samples.png')
    # plt.close()
    plt.show()

def plot_training_histories(custom_loss_history, epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), custom_loss_history, label='Custom NN Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    # plt.savefig('training_history.png')
    # plt.close()
    plt.show()

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load Fashion MNIST dataset
    print("Loading Fashion MNIST dataset...")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
    
    # Define class names for the Fashion MNIST dataset
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    # Convert labels to integers
    y = y.astype(int)
    
    # Sample a subset to make training faster (1000 samples)
    sample_size = 1000
    indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sampled = X[indices]
    y_sampled = y[indices]
    
    # Plot some sample images
    plot_sample_images(X_sampled, y_sampled, class_names)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled, test_size=0.2, random_state=42, stratify=y_sampled
    )
    
    # Preprocess the data
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # One-hot encode the targets for the custom neural network
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    
    print(f"Dataset shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Parameters for both models
    hidden_layer_sizes = (128, 64)  # Two hidden layers
    max_iter = 100
    learning_rate = 0.001
    
    # Train and evaluate the custom neural network
    print("\n=== Training Custom Neural Network ===")
    
    # Define the neural network architecture
    input_size = X_train_scaled.shape[1]  # Number of features
    output_size = len(np.unique(y_train))  # Number of classes
    
    layer_sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
    
    # Initialize the neural network
    custom_nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        hidden_layer_activation_functions=[relu] * len(hidden_layer_sizes),
        output_layer_activation_function=softmax,
        loss_function=categorical_cross_entropy,
        learning_rate=learning_rate,
        max_iter=max_iter,
        batch_size=32,
        optimizer="adam",
        l2_lambda=0.0001,  # Small L2 regularization
        initialization_method="he",
        verbose=True
    )
    
    # Start the timer for custom NN training
    start_time_custom = time.time()
    
    # Train the custom neural network
    custom_loss_history, _ = custom_nn.fit(X_train_scaled, y_train_onehot)
    
    # End the timer for custom NN training
    training_time_custom = time.time() - start_time_custom
    print(f"Custom Neural Network training time: {training_time_custom:.2f} seconds")
    
    # Predict with the custom neural network
    custom_predictions_proba = custom_nn.predict(X_test_scaled)
    custom_predictions = np.argmax(custom_predictions_proba, axis=1)
    
    # Calculate accuracy
    custom_accuracy = accuracy_score(y_test, custom_predictions)
    custom_cm = confusion_matrix(y_test, custom_predictions)
    
    print(f"Custom Neural Network Accuracy: {custom_accuracy:.4f}")
    
    # Train and evaluate sklearn's MLPClassifier
    print("\n=== Training Scikit-learn MLPClassifier ===")
    
    # Initialize MLPClassifier
    sklearn_mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 penalty parameter
        batch_size=32,
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        random_state=42,
        verbose=True
    )
    
    # Start the timer for scikit-learn training
    start_time_sklearn = time.time()
    
    # Train the MLPClassifier
    sklearn_mlp.fit(X_train_scaled, y_train)
    
    # End the timer for scikit-learn training
    training_time_sklearn = time.time() - start_time_sklearn
    print(f"Scikit-learn MLPClassifier training time: {training_time_sklearn:.2f} seconds")
    
    # Predict with the MLPClassifier
    sklearn_predictions = sklearn_mlp.predict(X_test_scaled)
    
    # Calculate accuracy
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    sklearn_cm = confusion_matrix(y_test, sklearn_predictions)
    
    print(f"Scikit-learn MLPClassifier Accuracy: {sklearn_accuracy:.4f}")
    
    # Plot confusion matrices
    plot_confusion_matrices(custom_cm, sklearn_cm, class_names)
    
    # Plot training history for custom NN
    plot_training_histories(custom_loss_history, max_iter)
    
    # Compare results
    print("\n=== Performance Comparison ===")
    print(f"Custom Neural Network Accuracy: {custom_accuracy:.4f}")
    print(f"Scikit-learn MLPClassifier Accuracy: {sklearn_accuracy:.4f}")
    print(f"Custom Neural Network Training Time: {training_time_custom:.2f} seconds")
    print(f"Scikit-learn MLPClassifier Training Time: {training_time_sklearn:.2f} seconds")
    
    # Print detailed classification reports
    print("\nCustom Neural Network Classification Report:")
    print(classification_report(y_test, custom_predictions, target_names=class_names))
    
    print("\nScikit-learn MLPClassifier Classification Report:")
    print(classification_report(y_test, sklearn_predictions, target_names=class_names))
    
    # Compare misclassifications
    mismatches = np.where(custom_predictions != sklearn_predictions)[0]
    print(f"\nNumber of samples where models disagree: {len(mismatches)}")
    
    if len(mismatches) > 0:
        # Analyze a few samples where predictions differ
        print("\nAnalysis of 5 random samples where predictions differ:")
        sample_mismatches = np.random.choice(mismatches, min(5, len(mismatches)), replace=False)
        
        for idx in sample_mismatches:
            true_label = y_test[idx]
            custom_pred = custom_predictions[idx]
            sklearn_pred = sklearn_predictions[idx]
            
            print(f"Sample {idx}:")
            print(f"  True label: {class_names[true_label]}")
            print(f"  Custom NN prediction: {class_names[custom_pred]}")
            print(f"  Scikit-learn prediction: {class_names[sklearn_pred]}")
            
            # Print top-3 probabilities from custom model
            top_probs_idx = np.argsort(custom_predictions_proba[idx])[::-1][:3]
            print("  Custom NN top probabilities:")
            for i, prob_idx in enumerate(top_probs_idx):
                print(f"    {i+1}. {class_names[prob_idx]}: {custom_predictions_proba[idx][prob_idx]:.4f}")
            print()

if __name__ == "__main__":
    main()