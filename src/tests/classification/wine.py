import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

from neural_network import *
# Load the custom neural network implementation from the provided code
# Note: The code assumes the NeuralNetwork class is defined in the current namespace
# along with all the activation and loss functions

# Load Wine dataset (a more challenging multi-class problem than Iris)
wine = load_wine()
X = wine.data
y = wine.target.reshape(-1, 1)
class_names = wine.target_names

# Add some noise to make it more challenging
np.random.seed(42)
noise_level = 0.5
X += np.random.normal(0, noise_level, X.shape)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode the target
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train_idx, y_test_idx = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
_, _, y_train_onehot, y_test_onehot = train_test_split(X_scaled, y_one_hot, test_size=0.3, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of classes: {len(class_names)}")

# Configuration
hidden_layer_sizes = (20, 10)  # Two hidden layers with 20 and 10 neurons
max_iter = 1000
layer_sizes = [X_train.shape[1]] + list(hidden_layer_sizes) + [y_train_onehot.shape[1]]

#############################################
# Scikit-learn MLPClassifier
#############################################
print("\nTraining scikit-learn MLPClassifier...")
start_time = time.time()

mlp_sklearn = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation='relu',
    solver='adam',
    alpha=0.0001,  # L2 regularization parameter
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=max_iter,
    random_state=42,
    verbose=False
)

mlp_sklearn.fit(X_train, y_train_idx.ravel())
sklearn_time = time.time() - start_time
print(f"Training completed in {sklearn_time:.2f} seconds")

# Evaluate scikit-learn model
y_pred_sklearn = mlp_sklearn.predict(X_test)
sklearn_accuracy = accuracy_score(y_test_idx, y_pred_sklearn)
print(f"Scikit-learn MLPClassifier accuracy: {sklearn_accuracy:.4f}")

#############################################
# Custom Neural Network Implementation
#############################################
print("\nTraining custom Neural Network...")
start_time = time.time()

custom_nn = NeuralNetwork(
    layer_sizes=layer_sizes,
    hidden_layer_activation_functions=[relu] * len(hidden_layer_sizes),
    output_layer_activation_function=softmax,
    loss_function=categorical_cross_entropy,
    learning_rate=0.001,
    max_iter=max_iter,
    batch_size=64,
    optimizer="adam",
    l2_lambda=0.0001,  # L2 regularization
    verbose=True
)

# Train the custom neural network
losses, final_gradient = custom_nn.fit(X_train, y_train_onehot)
custom_time = time.time() - start_time
print(f"Training completed in {custom_time:.2f} seconds")

# Make predictions with custom neural network
y_pred_custom_probs = custom_nn.predict(X_test)
y_pred_custom = np.argmax(y_pred_custom_probs, axis=1)
custom_accuracy = accuracy_score(y_test_idx, y_pred_custom)
print(f"Custom Neural Network accuracy: {custom_accuracy:.4f}")

#############################################
# Comparison and Visualization
#############################################
print("\n----- Performance Comparison -----")
print(f"Scikit-learn MLPClassifier: {sklearn_accuracy:.4f} (trained in {sklearn_time:.2f} seconds)")
print(f"Custom Neural Network: {custom_accuracy:.4f} (trained in {custom_time:.2f} seconds)")
print(f"Accuracy difference: {abs(sklearn_accuracy - custom_accuracy):.4f}")

# Classification report for both models
print("\nScikit-learn Classification Report:")
print(classification_report(y_test_idx, y_pred_sklearn, target_names=class_names))

print("\nCustom Neural Network Classification Report:")
print(classification_report(y_test_idx, y_pred_custom, target_names=class_names))

# Plot learning curve for custom neural network
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Custom Neural Network Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Visualize confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scikit-learn confusion matrix
cm_sklearn = confusion_matrix(y_test_idx, y_pred_sklearn)
sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Scikit-learn MLPClassifier\nConfusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Custom neural network confusion matrix
cm_custom = confusion_matrix(y_test_idx, y_pred_custom)
sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('Custom Neural Network\nConfusion Matrix')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Visualize decision boundaries for the first two features and first two classes
# (only if there are at least 3 classes, we'll use classes 0 and 1)
if len(class_names) >= 3:
    # Select only the first two features and the first two classes for visualization
    X_subset = X_scaled[:, :2]
    y_subset = y.ravel()
    mask = y_subset < 2
    X_subset = X_subset[mask]
    y_subset = y_subset[mask]
    
    # Create meshgrid
    h = 0.02  # step size in the mesh
    x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
    y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Create test points
    Z_points = np.c_[xx.ravel(), yy.ravel()]
    # Pad the test points with zeros for remaining features
    if X_scaled.shape[1] > 2:
        Z_points_padded = np.zeros((Z_points.shape[0], X_scaled.shape[1]))
        Z_points_padded[:, :2] = Z_points
    else:
        Z_points_padded = Z_points
    
    # Plot decision boundaries
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scikit-learn decision boundary
    Z_sklearn = mlp_sklearn.predict(Z_points_padded)
    Z_sklearn = Z_sklearn.reshape(xx.shape)
    axes[0].contourf(xx, yy, Z_sklearn, cmap=plt.cm.coolwarm, alpha=0.8)
    axes[0].scatter(X_subset[:, 0], X_subset[:, 1], c=y_subset, cmap=plt.cm.coolwarm, edgecolors='k')
    axes[0].set_title('Scikit-learn MLPClassifier\nDecision Boundary')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Custom neural network decision boundary
    Z_custom_probs = custom_nn.predict(Z_points_padded)
    Z_custom = np.argmax(Z_custom_probs, axis=1)
    Z_custom = Z_custom.reshape(xx.shape)
    axes[1].contourf(xx, yy, Z_custom, cmap=plt.cm.coolwarm, alpha=0.8)
    axes[1].scatter(X_subset[:, 0], X_subset[:, 1], c=y_subset, cmap=plt.cm.coolwarm, edgecolors='k')
    axes[1].set_title('Custom Neural Network\nDecision Boundary')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

print("\nComparison completed with visualizations displayed.")