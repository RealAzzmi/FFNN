import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from matplotlib.colors import ListedColormap

# Import the custom Neural Network implementation and its components
from neural_network import (NeuralNetwork, relu, sigmoid, tanh, softmax, 
                  categorical_cross_entropy, mean_squared_error, 
                  binary_cross_entropy)

# Set random seed for reproducibility
np.random.seed(42)

def generate_spiral_dataset(n_samples=300, n_classes=3, noise=0.5):
    """
    Generate a spiral dataset which is a challenging classification problem.
    
    Parameters:
    - n_samples: Number of samples per class
    - n_classes: Number of spiral arms (classes)
    - noise: Amount of noise to add
    
    Returns:
    - X: Features (2D coordinates)
    - y: Class labels
    """
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype=int)
    
    for j in range(n_classes):
        ix = range(n_samples * j, n_samples * (j + 1))
        r = np.linspace(0.0, 1, n_samples)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, n_samples) + np.random.randn(n_samples) * noise  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
        
    return X, y

def plot_spiral_data(X, y, title="Spiral Dataset"):
    """Plot the spiral dataset with colors for each class"""
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8, edgecolors='w')
    plt.title(title, fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(X, y, model, model_type="Custom NN", ax=None):
    """
    Plot the decision boundary for a model on 2D data.
    
    Parameters:
    - X: Feature matrix (2D)
    - y: Target vector
    - model: Trained model (either custom NN or scikit-learn's MLPClassifier)
    - model_type: String indicator of model type
    - ax: Matplotlib axis object (optional)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    # Set min and max for both axes with some padding
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    
    # Create a meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Flatten the grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Make predictions for each point in the meshgrid
    if model_type == "Custom NN":
        # For custom NN we need to consider its output structure (one-hot encoded)
        Z = model.predict(grid_points)
        Z = np.argmax(Z, axis=1)
    else:
        # For scikit-learn's MLPClassifier
        Z = model.predict(grid_points)
        
    # Reshape back to grid shape
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.8)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(f"{model_type} Decision Boundary", fontsize=14)
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    
    return ax

def main():
    print("Starting Neural Network Comparison on Spiral Dataset")
    
    # Step 1: Generate the spiral dataset
    print("\n1. Generating spiral dataset...")
    n_samples_per_class = 300
    n_classes = 3
    noise_level = 0.5
    
    X, y = generate_spiral_dataset(n_samples=n_samples_per_class, 
                                   n_classes=n_classes, 
                                   noise=noise_level)
    
    print(f"Dataset shape: {X.shape}, with {n_classes} classes")
    
    # Visualize the raw dataset
    # plot_spiral_data(X, y, title=f"Spiral Dataset ({n_classes} classes, noise={noise_level})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # One-hot encode targets for custom NN
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
    
    print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    
    # Step 2: Create deeper custom Neural Network (spiral data requires more complex model)
    print("\n2. Training custom Neural Network...")
    custom_nn = NeuralNetwork(
        layer_sizes=[X_train_scaled.shape[1], 50, 30, 20, n_classes],
        hidden_layer_activation_functions=[relu, tanh, relu],
        output_layer_activation_function=softmax,
        loss_function=categorical_cross_entropy,
        learning_rate=0.001,
        max_iter=2000,  # More iterations for this harder problem
        batch_size=32,
        optimizer="adam",
        l2_lambda=0.0001,  # Add regularization
        verbose=True
    )
    
    # Train and time custom NN
    custom_start_time = time.time()
    custom_loss_history, _ = custom_nn.fit(X_train_scaled, y_train_onehot)
    custom_train_time = time.time() - custom_start_time
    print(f"Custom NN training time: {custom_train_time:.2f} seconds")
    
    # Step 3: Train scikit-learn MLPClassifier with similar architecture
    print("\n3. Training scikit-learn MLPClassifier...")
    sklearn_nn = MLPClassifier(
        hidden_layer_sizes=(50, 30, 20),
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        learning_rate_init=0.001,
        max_iter=2000,
        batch_size=32,
        verbose=True,
        random_state=42
    )
    
    # Train and time scikit-learn NN
    sklearn_start_time = time.time()
    sklearn_nn.fit(X_train_scaled, y_train)
    sklearn_train_time = time.time() - sklearn_start_time
    print(f"Scikit-learn training time: {sklearn_train_time:.2f} seconds")
    
    # Get the loss history from scikit-learn MLP
    sklearn_loss_history = sklearn_nn.loss_curve_
    
    # Step 4: Make predictions
    print("\n4. Making predictions...")
    
    # Custom NN predictions
    custom_probs = custom_nn.predict(X_test_scaled)
    custom_predictions = np.argmax(custom_probs, axis=1)
    
    # Scikit-learn predictions
    sklearn_predictions = sklearn_nn.predict(X_test_scaled)
    
    # Step 5: Evaluate and compare metrics
    print("\n5. Comparing metrics...")
    
    # Accuracy
    custom_accuracy = accuracy_score(y_test, custom_predictions)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    print(f"Custom NN accuracy: {custom_accuracy:.4f}")
    print(f"Scikit-learn accuracy: {sklearn_accuracy:.4f}")
    
    # Classification reports
    print("\nCustom NN Classification Report:")
    print(classification_report(y_test, custom_predictions))
    
    print("\nScikit-learn Classification Report:")
    print(classification_report(y_test, sklearn_predictions))
    
    # Step 6: Visualize decision boundaries
    print("\n6. Creating decision boundary visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot custom NN decision boundary
    plot_decision_boundary(X_test, y_test, custom_nn, "Custom NN", ax=axes[0])
    
    # Plot scikit-learn decision boundary
    plot_decision_boundary(X_test, y_test, sklearn_nn, "Scikit-learn NN", ax=axes[1])
    
    plt.suptitle(f"Decision Boundaries for Spiral Dataset ({n_classes} classes)", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Step 7: Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(custom_loss_history, label='Custom NN')
    
    # Adjust sklearn loss to match the scale if needed
    if max(sklearn_loss_history) > 10 * max(custom_loss_history):
        scaled_sklearn_loss = [x / max(sklearn_loss_history) * max(custom_loss_history) for x in sklearn_loss_history]
        plt.plot(scaled_sklearn_loss, label='Scikit-learn MLP (scaled)')
    else:
        plt.plot(sklearn_loss_history, label='Scikit-learn MLP')
        
    plt.title('Learning Curves', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Step 8: Plot confusion matrices
    from sklearn.metrics import ConfusionMatrixDisplay
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Custom NN confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, custom_predictions, 
        ax=axes[0], 
        colorbar=False,
        cmap='Blues'
    )
    axes[0].set_title("Custom NN Confusion Matrix", fontsize=14)
    
    # Scikit-learn confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, sklearn_predictions, 
        ax=axes[1], 
        colorbar=False,
        cmap='Blues'
    )
    axes[1].set_title("Scikit-learn NN Confusion Matrix", fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    # Step 9: Summary and conclusion
    print("\n8. Summary Comparison:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Custom NN':<15} {'Scikit-learn':<15}")
    print("-" * 50)
    print(f"{'Training time (s)':<25} {custom_train_time:<15.2f} {sklearn_train_time:<15.2f}")
    print(f"{'Accuracy':<25} {custom_accuracy:<15.4f} {sklearn_accuracy:<15.4f}")
    print("-" * 50)
    
    # Calculate performance ratios
    time_ratio = custom_train_time / sklearn_train_time
    accuracy_ratio = custom_accuracy / sklearn_accuracy
    print(f"Custom NN is {time_ratio:.2f}x {'slower' if time_ratio > 1 else 'faster'} in training than scikit-learn")
    print(f"Custom NN accuracy is {accuracy_ratio:.2f}x that of scikit-learn")
    
    # Conclusion
    print("\nConclusion:")
    if custom_accuracy > sklearn_accuracy:
        print("The custom Neural Network implementation achieved higher accuracy on the spiral dataset.")
    elif custom_accuracy < sklearn_accuracy:
        print("Scikit-learn's MLPClassifier achieved higher accuracy on the spiral dataset.")
    else:
        print("Both implementations achieved the same accuracy on the spiral dataset.")
    
    print("The spiral dataset is particularly challenging because it requires non-linear decision boundaries.")
    print("This test demonstrates how well both implementations can handle complex classification tasks.")

if __name__ == "__main__":
    main()