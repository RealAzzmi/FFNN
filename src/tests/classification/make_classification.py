import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import seaborn as sns

# Import your custom neural network implementation
from neural_network import NeuralNetwork, sigmoid, relu, softmax, categorical_cross_entropy, tanh

# Set random seed for reproducibility
np.random.seed(42)

# Generate a challenging multi-class classification dataset
def generate_dataset(n_samples=1000, n_classes=4, n_features=20, n_informative=10):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.1,  # Add some noise
        weights=None,
        random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoding for custom NN
    y_train_one_hot = np.zeros((y_train.size, n_classes))
    y_train_one_hot[np.arange(y_train.size), y_train] = 1
    
    y_test_one_hot = np.zeros((y_test.size, n_classes))
    y_test_one_hot[np.arange(y_test.size), y_test] = 1
    
    return X_train, X_test, y_train, y_train_one_hot, y_test, y_test_one_hot

# Function to train and evaluate the custom neural network
def train_custom_nn(X_train, y_train_one_hot, X_test, y_test_one_hot, hidden_layer_sizes, max_iter=500):
    # Get input and output dimensions
    n_features = X_train.shape[1]
    n_classes = y_train_one_hot.shape[1]
    
    # Define layer sizes including input and output layers
    layer_sizes = [n_features] + list(hidden_layer_sizes) + [n_classes]
    
    # Initialize activation functions for hidden layers
    hidden_activations = [relu] * len(hidden_layer_sizes)
    
    # Create and train the custom neural network
    start_time = time.time()
    
    custom_nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        hidden_layer_activation_functions=hidden_activations,
        output_layer_activation_function=softmax,
        loss_function=categorical_cross_entropy,
        learning_rate=0.01,
        max_iter=max_iter,
        batch_size=32,
        optimizer="adam",
        l2_lambda=0.0001,  # Light regularization
        verbose=True
    )
    
    # Train the network
    train_losses = custom_nn.fit(X_train, y_train_one_hot)
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    y_pred_probs = custom_nn.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_orig = np.argmax(y_test_one_hot, axis=1)
    
    accuracy = accuracy_score(y_test_orig, y_pred)
    conf_matrix = confusion_matrix(y_test_orig, y_pred)
    
    return custom_nn, train_losses, accuracy, conf_matrix, training_time, y_pred, y_test_orig

# Function to train and evaluate scikit-learn's MLPClassifier
def train_sklearn_mlp(X_train, y_train, X_test, y_test, hidden_layer_sizes, max_iter=500):
    start_time = time.time()
    
    # Create and train the scikit-learn MLP
    sklearn_mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        batch_size=32,
        learning_rate='adaptive',
        max_iter=max_iter,
        random_state=42,
        verbose=True
    )
    
    # Train the model
    sklearn_mlp.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    y_pred = sklearn_mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return sklearn_mlp, accuracy, conf_matrix, training_time, y_pred

# Function to plot confusion matrices
def plot_confusion_matrices(conf_matrix_custom, conf_matrix_sklearn, n_classes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(conf_matrix_custom, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Custom Neural Network Confusion Matrix')
    ax1.set_xlabel('Predicted Labels')
    ax1.set_ylabel('True Labels')
    
    sns.heatmap(conf_matrix_sklearn, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('scikit-learn MLPClassifier Confusion Matrix')
    ax2.set_xlabel('Predicted Labels')
    ax2.set_ylabel('True Labels')
    
    plt.tight_layout()
    plt.show()

# Function to plot loss curve
def plot_loss_curve(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss Curve - Custom Neural Network')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

# Function to create a simple 2D visualization of class distributions
def plot_feature_importance(X_test, y_test, title):
    plt.figure(figsize=(12, 6))
    
    # Choose two features with highest variance
    feature_vars = np.var(X_test, axis=0)
    top_features = np.argsort(-feature_vars)[:2]
    
    plt.scatter(X_test[:, top_features[0]], X_test[:, top_features[1]], c=y_test, cmap='viridis', 
                alpha=0.7, edgecolors='k', s=50)
    
    plt.colorbar(label='Class')
    plt.xlabel(f'Feature {top_features[0]}')
    plt.ylabel(f'Feature {top_features[1]}')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Function to plot class prediction distributions
def plot_prediction_distribution(custom_preds, sklearn_preds, true_labels, n_classes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot distribution of predictions for custom NN
    custom_dist = [np.sum(custom_preds == i) for i in range(n_classes)]
    true_dist = [np.sum(true_labels == i) for i in range(n_classes)]
    
    x = np.arange(n_classes)
    width = 0.35
    
    ax1.bar(x - width/2, true_dist, width, label='True Distribution', color='green', alpha=0.7)
    ax1.bar(x + width/2, custom_dist, width, label='Predicted Distribution', color='blue', alpha=0.7)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title('Custom Neural Network Class Distribution')
    ax1.set_xticks(x)
    ax1.legend()
    
    # Plot distribution of predictions for sklearn
    sklearn_dist = [np.sum(sklearn_preds == i) for i in range(n_classes)]
    
    ax2.bar(x - width/2, true_dist, width, label='True Distribution', color='green', alpha=0.7)
    ax2.bar(x + width/2, sklearn_dist, width, label='Predicted Distribution', color='orange', alpha=0.7)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_title('scikit-learn MLPClassifier Class Distribution')
    ax2.set_xticks(x)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Main function to run the comparison
def run_comparison():
    print("Generating dataset...")
    X_train, X_test, y_train, y_train_one_hot, y_test, y_test_one_hot = generate_dataset(
        n_samples=2000, n_classes=4, n_features=20, n_informative=10
    )
    
    n_classes = y_train_one_hot.shape[1]
    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}, Classes: {n_classes}")
    
    # Create a simple visualization of the dataset
    # plot_feature_importance(X_test, y_test, 'Dataset Visualization (Top 2 Features by Variance)')
    
    # Define hidden layer architecture - using smaller layers for faster execution
    hidden_layer_sizes = (32, 16)
    max_iter = 200  # Reduced for faster execution
    
    # Train and evaluate custom neural network
    print("\nTraining custom neural network...")
    custom_nn, train_losses, custom_accuracy, custom_conf_matrix, custom_time, custom_pred, y_test_orig = train_custom_nn(
        X_train, y_train_one_hot, X_test, y_test_one_hot, hidden_layer_sizes, max_iter
    )
    
    # Train and evaluate scikit-learn MLPClassifier
    print("\nTraining scikit-learn MLPClassifier...")
    sklearn_mlp, sklearn_accuracy, sklearn_conf_matrix, sklearn_time, sklearn_pred = train_sklearn_mlp(
        X_train, y_train, X_test, y_test, hidden_layer_sizes, max_iter
    )
    
    # Print results
    print("\n----- Results -----")
    print(f"Custom Neural Network Accuracy: {custom_accuracy:.4f}")
    print(f"scikit-learn MLPClassifier Accuracy: {sklearn_accuracy:.4f}")
    print(f"Custom Neural Network Training Time: {custom_time:.2f} seconds")
    print(f"scikit-learn MLPClassifier Training Time: {sklearn_time:.2f} seconds")
    
    # Plot confusion matrices
    plot_confusion_matrices(custom_conf_matrix, sklearn_conf_matrix, n_classes)
    
    # Plot loss curve for custom neural network
    plot_loss_curve(train_losses)
    
    # Plot class prediction distributions
    plot_prediction_distribution(custom_pred, sklearn_pred, y_test_orig, n_classes)
    
    # Print detailed classification report
    print("\nClassification Report - Custom Neural Network:")
    print(classification_report(y_test_orig, custom_pred))
    
    print("\nClassification Report - scikit-learn MLPClassifier:")
    print(classification_report(y_test, sklearn_pred))
    
    # Compare misclassifications
    custom_misclass = np.sum(custom_pred != y_test_orig)
    sklearn_misclass = np.sum(sklearn_pred != y_test)
    
    print(f"\nCustom Neural Network Total Misclassifications: {custom_misclass} out of {len(y_test)}")
    print(f"scikit-learn MLPClassifier Total Misclassifications: {sklearn_misclass} out of {len(y_test)}")
    
    # Count common misclassifications
    common_misclass = np.sum((custom_pred != y_test_orig) & (sklearn_pred != y_test))
    print(f"Common Misclassifications (both models wrong): {common_misclass}")
    
    # Calculate per-class accuracies
    custom_class_acc = []
    sklearn_class_acc = []
    
    for cls in range(n_classes):
        cls_indices = (y_test_orig == cls)
        custom_class_acc.append(np.mean(custom_pred[cls_indices] == y_test_orig[cls_indices]))
        sklearn_class_acc.append(np.mean(sklearn_pred[cls_indices] == y_test[cls_indices]))
    
    print("\nPer-Class Accuracies:")
    for cls in range(n_classes):
        print(f"Class {cls}: Custom NN = {custom_class_acc[cls]:.4f}, scikit-learn = {sklearn_class_acc[cls]:.4f}")

if __name__ == "__main__":
    run_comparison()