
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.patches as patches

def plot_weight_distribution(weights, layers=None):
    if layers is None:
        layers = list(range(1, len(weights)))
    
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 4*len(layers)))
    if len(layers) == 1:
        axes = [axes]
    
    fig.suptitle('Weight Distributions per Layer', fontsize=16)
    for i, layer in enumerate(layers):
        layer_weights = weights[layer]
        
        if layer_weights is not None:
            flat_weights = layer_weights.flatten()
            mean = np.mean(flat_weights)
            median = np.median(flat_weights)
            std = np.std(flat_weights)
            min_val = np.min(flat_weights)
            max_val = np.max(flat_weights)

            axes[i].hist(layer_weights.flatten(), bins=50, color='blue', alpha=0.7)
            axes[i].set_title(f'Layer {layer} Weight Distribution')
            axes[i].set_xlabel('Weight Values')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[i].text(0.5, 0.5, 'No Weights', 
                         horizontalalignment='center', 
                         verticalalignment='center')
    plt.tight_layout()
    plt.show()

def plot_weight_gradient_distribution(gradients, layers=None):
    if layers is None:
        layers = list(range(1, len(gradients)))
    
    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 4*len(layers)))
    if len(layers) == 1:
        axes = [axes]
    
    fig.suptitle('Weight Gradient Distributions per Layer', fontsize=16)
    for i, layer in enumerate(layers):
        layer_gradients = gradients[layer]
        
        if layer_gradients is not None:
            flat_gradients = layer_gradients.flatten()
            mean = np.mean(flat_gradients)
            median = np.median(flat_gradients)
            std = np.std(flat_gradients)
            min_val = np.min(flat_gradients)
            max_val = np.max(flat_gradients)
            iqr = np.percentile(flat_gradients, 75) - np.percentile(flat_gradients, 25)
            bin_width = max(2 * iqr / (len(flat_gradients) ** (1/3)), 1e-6)
            data_range = np.max(flat_gradients) - np.min(flat_gradients)
            num_bins = max(int(data_range / bin_width), 50)
            axes[i].hist(flat_gradients, bins=num_bins, color='red', alpha=0.7, edgecolor='black')

            axes[i].hist(layer_gradients.flatten(), bins=50, color='red', alpha=0.7)
            axes[i].axvline(mean, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.4f}')
            axes[i].axvline(median, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median:.4f}')
            axes[i].set_title(f'Layer {layer} Weight Gradient Distribution\n'
                               f'Mean: {mean:.4f}, Median: {median:.4f}, Std Dev: {std:.4f}', 
                               fontsize=10)
            axes[i].set_xlabel('Gradient Values')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[i].text(0.5, 0.5, 'No Gradients', 
                         horizontalalignment='center', 
                         verticalalignment='center')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(conf_matrix_custom, conf_matrix_sklearn):
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

def plot_loss_curve(train_losses, validation_losses=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, color='blue', label='Training Loss')

    if validation_losses:
        plt.plot(validation_losses, color='red', label='Validation Loss')

    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_decision_boundaries(X, y, X_test, y_test, models, model_names):
    """
    Plot decision boundaries for multiple models.
    
    Parameters:
    X : feature data
    y : target labels
    models : list of trained models with predict method
    model_names : list of model names for the legend
    """
    # Set up the plot
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    if len(models) == 1:
        axes = [axes]
    
    # Define the mesh grid for plotting
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Define color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    
    # Plot each model
    for idx, (model, model_name) in enumerate(zip(models, model_names)):
        # Predict on the mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary and training points
        axes[idx].contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=20)
        axes[idx].set_xlim(xx.min(), xx.max())
        axes[idx].set_ylim(yy.min(), yy.max())
        axes[idx].set_title(f"{model_name}")
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

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