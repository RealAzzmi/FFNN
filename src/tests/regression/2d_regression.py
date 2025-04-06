import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time

# Import the custom neural network implementation
from neural_network import NeuralNetwork, relu, tanh, sigmoid, linear, mean_squared_error

# Create a complex 2D regression dataset with multiple overlapping functions
def create_complex_regression_dataset(n_samples=1000, noise=0.4):
    np.random.seed(42)
    
    # Generate uniformly distributed points in 2D space
    X = np.random.uniform(-4, 4, (n_samples, 2))
    
    # Create a complex target function with multiple components
    y = np.zeros((n_samples, 1))
    
    # Component 1: Spiral pattern
    radius = np.sqrt(X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)
    theta = np.arctan2(X[:, 1], X[:, 0]).reshape(-1, 1)
    y += 0.5 * np.sin(3 * radius + 5 * theta)
    
    # Component 2: Radial waves
    y += 0.3 * np.sin(2 * radius) * np.cos(3 * theta)
    
    # Component 3: Exponential decay from center
    y += 0.4 * np.exp(-0.2 * radius)
    
    # Component 4: Gaussian bumps
    centers = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
    for cx, cy in centers:
        dist = np.sqrt((X[:, 0] - cx)**2 + (X[:, 1] - cy)**2).reshape(-1, 1)
        y += 0.3 * np.exp(-dist**2)
    
    # Add random noise
    y += noise * np.random.randn(n_samples, 1)
    
    return X, y

def plot_regression_surface(X, y, model, title, ax):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on the mesh grid
    Z = np.c_[xx.ravel(), yy.ravel()]
    
    if isinstance(model, NeuralNetwork):
        Z_pred = np.array([model.predict(z).flatten() for z in Z])
    else:
        Z_pred = model.predict(Z).reshape(-1)
        
    Z_pred = Z_pred.reshape(xx.shape)
    
    # Plot the surface
    surf = ax.contourf(xx, yy, Z_pred, cmap=plt.cm.viridis, alpha=0.8)
    plt.colorbar(surf, ax=ax)
    
    # Plot the training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.viridis, 
               edgecolor='k', s=20)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)

def main():
    # Generate the dataset
    X, y = create_complex_regression_dataset(n_samples=1500, noise=0.2)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.viridis, edgecolor='k')
    plt.colorbar()
    plt.title('Complex Regression Dataset')
    plt.tight_layout()
    # plt.savefig('complex_regression_dataset.png')
    plt.close()
    
    # Train scikit-learn MLPRegressor
    print("\nTraining scikit-learn MLPRegressor...")
    start_time = time.time()
    
    sklearn_mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization parameter
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        verbose=True
    )
    
    sklearn_mlp.fit(X_train_scaled, y_train.ravel())
    sklearn_time = time.time() - start_time
    print(f"scikit-learn training time: {sklearn_time:.2f} seconds")
    
    # Evaluate scikit-learn model
    y_pred_sklearn = sklearn_mlp.predict(X_test_scaled).reshape(-1, 1)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    
    print(f"scikit-learn MLPRegressor - MSE: {mse_sklearn:.6f}, R²: {r2_sklearn:.6f}")
    
    # Train custom NeuralNetwork
    print("\nTraining custom NeuralNetwork...")
    start_time = time.time()
    
    # Define the architecture of the custom neural network
    input_size = X_train_scaled.shape[1]
    custom_nn = NeuralNetwork(
        layer_sizes=[input_size, 64, 32, 16, 1],  # Same architecture as scikit-learn
        hidden_layer_activation_functions=[relu, relu, relu],  # relu for hidden layers
        output_layer_activation_function=linear,  # Linear for regression output
        loss_function=mean_squared_error,
        learning_rate=0.001,
        max_iter=500,
        batch_size=64,
        optimizer="adam",  # Use Adam optimizer
        l2_lambda=0.0001,  # L2 regularization
        verbose=True
    )
    
    # Train the custom neural network
    losses = custom_nn.fit(X_train_scaled, y_train)
    custom_time = time.time() - start_time
    print(f"Custom neural network training time: {custom_time:.2f} seconds")
    
    # Evaluate custom model
    y_pred_custom = np.array([custom_nn.predict(x).flatten() for x in X_test_scaled])
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    r2_custom = r2_score(y_test, y_pred_custom)
    
    print(f"Custom NeuralNetwork - MSE: {mse_custom:.6f}, R²: {r2_custom:.6f}")
    
    # Plot the training loss curve for custom model
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss for Custom Neural Network')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    # plt.savefig('custom_nn_loss.png')
    plt.close()
    
    # Plot the prediction surfaces
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    plot_regression_surface(X_test, y_test, sklearn_mlp, 
                            f'scikit-learn MLPRegressor\nMSE: {mse_sklearn:.6f}, R²: {r2_sklearn:.6f}', 
                            axes[0])
    
    plot_regression_surface(X_test, y_test, custom_nn, 
                            f'Custom NeuralNetwork\nMSE: {mse_custom:.6f}, R²: {r2_custom:.6f}', 
                            axes[1])
    
    plt.tight_layout()
    # plt.savefig('regression_comparison.png')
    plt.show()
    
    # Print a comparison summary
    print("\nPerformance Comparison:")
    print(f"{'Model':<25} {'MSE':<15} {'R²':<15} {'Training Time (s)':<20}")
    print("-" * 75)
    print(f"{'scikit-learn MLPRegressor':<25} {mse_sklearn:<15.6f} {r2_sklearn:<15.6f} {sklearn_time:<20.2f}")
    print(f"{'Custom NeuralNetwork':<25} {mse_custom:<15.6f} {r2_custom:<15.6f} {custom_time:<20.2f}")
    
    # Calculate the performance difference
    mse_diff = ((mse_custom - mse_sklearn) / mse_sklearn) * 100
    r2_diff = ((r2_custom - r2_sklearn) / r2_sklearn) * 100 if r2_sklearn != 0 else float('inf')
    time_diff = ((custom_time - sklearn_time) / sklearn_time) * 100
    
    print("\nPerformance Differences (Custom vs scikit-learn):")
    print(f"MSE: {'higher' if mse_diff > 0 else 'lower'} by {abs(mse_diff):.2f}%")
    print(f"R²: {'higher' if r2_diff > 0 else 'lower'} by {abs(r2_diff):.2f}%")
    print(f"Training time: {'longer' if time_diff > 0 else 'shorter'} by {abs(time_diff):.2f}%")
    
    # Try with some parameter tuning for custom NN if it's performing significantly worse
    if mse_custom > 1.2 * mse_sklearn:
        print("\nTrying custom NeuralNetwork with tuned parameters...")
        
        custom_nn_tuned = NeuralNetwork(
            layer_sizes=[input_size, 100, 50, 25, 1],  # Larger architecture
            hidden_layer_activation_functions=[tanh, tanh, tanh],  # Try tanh instead of relu
            output_layer_activation_function=linear,
            loss_function=mean_squared_error,
            learning_rate=0.0005,  # Lower learning rate
            max_iter=800,  # More iterations
            batch_size=32,  # Smaller batch size
            optimizer="adam",
            l2_lambda=0.00005,  # Adjusted regularization
            verbose=True
        )
        
        start_time = time.time()
        losses = custom_nn_tuned.fit(X_train_scaled, y_train)
        custom_tuned_time = time.time() - start_time
        
        y_pred_custom_tuned = np.array([custom_nn_tuned.predict(x).flatten() for x in X_test_scaled])
        mse_custom_tuned = mean_squared_error(y_test, y_pred_custom_tuned)
        r2_custom_tuned = r2_score(y_test, y_pred_custom_tuned)
        
        print(f"Tuned Custom NeuralNetwork - MSE: {mse_custom_tuned:.6f}, R²: {r2_custom_tuned:.6f}")
        print(f"Training time: {custom_tuned_time:.2f} seconds")
        
        # Plot with tuned model
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        plot_regression_surface(X_test, y_test, sklearn_mlp, 
                                f'scikit-learn MLPRegressor\nMSE: {mse_sklearn:.6f}, R²: {r2_sklearn:.6f}', 
                                axes[0])
        
        plot_regression_surface(X_test, y_test, custom_nn, 
                                f'Custom NeuralNetwork\nMSE: {mse_custom:.6f}, R²: {r2_custom:.6f}', 
                                axes[1])
        
        plot_regression_surface(X_test, y_test, custom_nn_tuned, 
                                f'Tuned Custom NeuralNetwork\nMSE: {mse_custom_tuned:.6f}, R²: {r2_custom_tuned:.6f}', 
                                axes[2])
        
        plt.tight_layout()
        # plt.savefig('regression_comparison_with_tuned.png')
        plt.show()

if __name__ == "__main__":
    main()