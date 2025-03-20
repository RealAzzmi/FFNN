
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

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
        axes[idx].set_title(f"{model_name}\nAccuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
