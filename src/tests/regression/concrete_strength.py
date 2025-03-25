import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Import your neural network implementation
from neural_network import NeuralNetwork, mean_squared_error, relu, tanh, sigmoid, linear

def load_concrete_data():
    """
    Load the Concrete Compressive Strength dataset.
    
    The dataset contains 8 input features related to concrete composition and age,
    and the target variable is the concrete compressive strength.
    
    Features:
    1. Cement
    2. Blast Furnace Slag
    3. Fly Ash
    4. Water
    5. Superplasticizer
    6. Coarse Aggregate
    7. Fine Aggregate
    8. Age
    
    Target:
    Concrete Compressive Strength (MPa)
    """
    # URL for concrete strength dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/concrete.data'
    
    # Column names for the dataset
    column_names = [
        'Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water',
        'Superplasticizer', 'CoarseAggregate', 'FineAggregate',
        'Age', 'ConcreteStrength'
    ]
    
    try:
        # Try to load the dataset from the URL
        data = pd.read_csv(url, header=None, names=column_names)
    except Exception as e:
        print(f"Failed to load from URL: {e}")
        print("Using simulated data instead...")
        
        # Create simulated data if URL load fails
        np.random.seed(42)
        n_samples = 1030
        
        # Reasonable ranges for each feature based on domain knowledge
        cement = np.random.uniform(100, 500, n_samples)
        blast_furnace_slag = np.random.uniform(0, 350, n_samples)
        fly_ash = np.random.uniform(0, 200, n_samples)
        water = np.random.uniform(100, 250, n_samples)
        superplasticizer = np.random.uniform(0, 20, n_samples)
        coarse_aggregate = np.random.uniform(700, 1200, n_samples)
        fine_aggregate = np.random.uniform(500, 1000, n_samples)
        age = np.random.choice([1, 3, 7, 14, 28, 56, 90, 180, 365], n_samples)
        
        # Generate target using a non-linear formula with some noise
        # This is a simplified model of concrete strength
        strength = (
            0.2 * cement +
            0.1 * blast_furnace_slag +
            0.05 * fly_ash -
            0.3 * water +
            1.0 * superplasticizer +
            0.01 * coarse_aggregate +
            0.01 * fine_aggregate +
            0.3 * np.log(age + 1)
        ) + np.random.normal(0, 5, n_samples)
        
        # Ensure strength is positive
        strength = np.maximum(strength, 1)
        
        # Create the dataframe
        data = pd.DataFrame({
            'Cement': cement,
            'BlastFurnaceSlag': blast_furnace_slag,
            'FlyAsh': fly_ash,
            'Water': water,
            'Superplasticizer': superplasticizer,
            'CoarseAggregate': coarse_aggregate,
            'FineAggregate': fine_aggregate,
            'Age': age,
            'ConcreteStrength': strength
        })
    
    return data

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate the model and print metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return mse, rmse, r2

def plot_predictions(y_true, y_pred_custom, y_pred_sklearn, title):
    """Plot actual vs predicted values for both models."""
    plt.figure(figsize=(12, 6))
    
    # Sort for better visualization
    sorted_indices = np.argsort(y_true)
    y_true_sorted = y_true[sorted_indices]
    y_pred_custom_sorted = y_pred_custom[sorted_indices]
    y_pred_sklearn_sorted = y_pred_sklearn[sorted_indices]
    
    # Plot actual values
    plt.plot(range(len(y_true_sorted)), y_true_sorted, 'o-', label='Actual Values', alpha=0.7)
    
    # Plot custom NN predictions
    plt.plot(range(len(y_pred_custom_sorted)), y_pred_custom_sorted, 'x-', label='Custom NN Predictions', alpha=0.7)
    
    # Plot sklearn predictions
    plt.plot(range(len(y_pred_sklearn_sorted)), y_pred_sklearn_sorted, 's-', label='Sklearn Predictions', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Sample Index (Sorted by True Value)')
    plt.ylabel('Concrete Strength (MPa)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('concrete_predictions_comparison.png')
    plt.show()

def plot_training_history(custom_losses, sklearn_losses, title):
    """Plot training loss over iterations for both models."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(custom_losses, label='Custom NN Loss')
    plt.plot(sklearn_losses, label='Sklearn Loss')
    
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often helps visualize convergence
    plt.tight_layout()
    # plt.savefig('training_loss_comparison.png')
    plt.show()

def plot_regression_comparison(y_true, y_pred_custom, y_pred_sklearn):
    """Plot regression comparison as scatter plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Custom NN
    ax1.scatter(y_true, y_pred_custom, alpha=0.5)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax1.set_xlabel('Actual Strength (MPa)')
    ax1.set_ylabel('Predicted Strength (MPa)')
    ax1.set_title('Custom Neural Network')
    ax1.grid(True, alpha=0.3)
    
    # Sklearn
    ax2.scatter(y_true, y_pred_sklearn, alpha=0.5)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax2.set_xlabel('Actual Strength (MPa)')
    ax2.set_ylabel('Predicted Strength (MPa)')
    ax2.set_title('Sklearn MLPRegressor')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.savefig('regression_comparison.png')
    plt.show()

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load the concrete strength dataset
    print("Loading Concrete Strength dataset...")
    data = load_concrete_data()
    
    # Display basic dataset information
    print(f"Dataset shape: {data.shape}")
    print("\nFeature statistics:")
    print(data.describe())
    
    # Prepare features and target
    X = data.iloc[:, :-1].values  # All columns except the last one
    y = data.iloc[:, -1].values   # Last column is the target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale the target for better neural network performance
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    print("\nTraining set shape:", X_train_scaled.shape)
    print("Test set shape:", X_test_scaled.shape)
    
    # --------------------------
    # Custom Neural Network
    # --------------------------
    print("\n\nTraining Custom Neural Network...")
    
    # Define network architecture
    input_size = X_train_scaled.shape[1]
    layer_sizes = [input_size, 16, 8, 1]
    
    # Create and train the custom neural network
    start_time = time.time()
    
    # Use of tanh activation for hidden layers and linear for output
    # since this is a regression problem
    custom_nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        hidden_layer_activation_functions=[tanh, tanh],  # One for each hidden layer
        output_layer_activation_function=linear,  # Linear activation for regression
        loss_function=mean_squared_error,
        learning_rate=0.01,
        max_iter=1000,
        batch_size=32,
        optimizer="adam",
        l2_lambda=0.0001,  # Light regularization
        verbose=True
    )
    
    # Reshape data for single samples in forward_propagation
    custom_losses = custom_nn.fit(X_train_scaled, y_train_scaled.reshape(-1, 1))
    
    custom_train_time = time.time() - start_time
    print(f"Custom NN training time: {custom_train_time:.2f} seconds")
    
    # Make predictions
    custom_pred_scaled = []
    for i in range(X_test_scaled.shape[0]):
        pred = custom_nn.predict(X_test_scaled[i])
        custom_pred_scaled.append(pred.flatten()[0])
    
    custom_pred_scaled = np.array(custom_pred_scaled)
    
    # Inverse transform the predictions back to original scale
    custom_pred = y_scaler.inverse_transform(custom_pred_scaled.reshape(-1, 1)).flatten()
    
    # --------------------------
    # Sklearn MLPRegressor
    # --------------------------
    print("\n\nTraining Sklearn MLPRegressor...")
    
    # Create and train sklearn's MLPRegressor with similar architecture
    start_time = time.time()
    
    sklearn_nn = MLPRegressor(
        hidden_layer_sizes=(16, 8),  # Same hidden layers as custom NN
        activation='tanh',           # Same activation
        solver='adam',               # Same optimizer
        alpha=0.0001,                # L2 regularization equivalent
        batch_size=32,
        learning_rate_init=0.01,
        max_iter=1000,
        random_state=42,
        verbose=True
    )
    
    # Fit the sklearn model
    sklearn_nn.fit(X_train_scaled, y_train_scaled)
    
    sklearn_train_time = time.time() - start_time
    print(f"Sklearn training time: {sklearn_train_time:.2f} seconds")
    
    # Get loss curve from sklearn
    sklearn_losses = sklearn_nn.loss_curve_
    
    # Make predictions
    sklearn_pred_scaled = sklearn_nn.predict(X_test_scaled)
    
    # Inverse transform the predictions back to original scale
    sklearn_pred = y_scaler.inverse_transform(sklearn_pred_scaled.reshape(-1, 1)).flatten()
    
    # --------------------------
    # Evaluation and Comparison
    # --------------------------
    print("\n===== MODEL EVALUATION =====")
    
    # Evaluate the models
    custom_metrics = evaluate_model(y_test, custom_pred, "Custom Neural Network")
    sklearn_metrics = evaluate_model(y_test, sklearn_pred, "Sklearn MLPRegressor")
    
    # Print training time comparison
    print("\nTraining Time Comparison:")
    print(f"Custom NN: {custom_train_time:.2f} seconds")
    print(f"Sklearn: {sklearn_train_time:.2f} seconds")
    print(f"Ratio (Custom/Sklearn): {custom_train_time/sklearn_train_time:.2f}x")
    
    # Calculate percentage difference in metrics
    mse_diff = (custom_metrics[0] - sklearn_metrics[0]) / sklearn_metrics[0] * 100
    r2_diff = (custom_metrics[2] - sklearn_metrics[2]) / sklearn_metrics[2] * 100 if sklearn_metrics[2] != 0 else float('inf')
    
    print("\nPerformance Difference (Custom vs Sklearn):")
    print(f"MSE Difference: {mse_diff:.2f}% ({'higher' if mse_diff > 0 else 'lower'} than Sklearn)")
    print(f"R² Difference: {r2_diff:.2f}% ({'higher' if r2_diff > 0 else 'lower'} than Sklearn)")
    
    # --------------------------
    # Visualization
    # --------------------------
    # Plot actual vs predicted values
    plot_predictions(y_test, custom_pred, sklearn_pred, 
                     'Concrete Strength: Actual vs Predicted')
    
    # Plot training history
    plot_training_history(custom_losses, sklearn_losses, 
                          'Training Loss Over Iterations')
    
    # Plot regression comparison
    plot_regression_comparison(y_test, custom_pred, sklearn_pred)
    
    print("\nTesting complete! Visualization plots have been saved.")

if __name__ == "__main__":
    main()