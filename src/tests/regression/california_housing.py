import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import the custom neural network implementation
from neural_network import NeuralNetwork, mean_squared_error, relu, linear

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(sample_size=2000):
    """Load California Housing dataset and preprocess it with a limited sample size.
    
    Args:
        sample_size: Number of samples to use from the dataset
    """
    print(f"Loading California Housing dataset (sample size: {sample_size})...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Use only a sample of the data
    if sample_size and sample_size < len(X):
        indices = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale the target values
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"Data shape: X_train: {X_train_scaled.shape}, y_train: {y_train_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler

def evaluate_model(model, X_test, y_test, y_scaler=None, model_name="Model"):
    """Evaluate the model and print metrics."""
    start_time = time.time()
    
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        # For custom NN that might not handle batched input well
        y_pred = np.array([model.predict(X_test[i].reshape(1, -1)).flatten() for i in range(X_test.shape[0])])
    
    # If we used scaled targets, transform back
    if y_scaler is not None:
        y_pred_original = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_pred_original = y_pred
        y_test_original = y_test
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)
    
    prediction_time = time.time() - start_time
    
    print(f"\n{model_name} Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    
    return y_pred_original, mse, rmse, r2, prediction_time

def test_custom_nn():
    """Test the custom neural network implementation."""
    # Load and preprocess data (already sampled)
    X_train, X_test, y_train, y_test, y_scaler = load_and_preprocess_data()
    
    # Train the custom neural network
    print("\nTraining Custom Neural Network...")
    start_time = time.time()
    
    input_size = X_train.shape[1]
    custom_nn = NeuralNetwork(
        layer_sizes=[input_size, 64, 32, 1],
        hidden_layer_activation_functions=[relu, relu],
        output_layer_activation_function=linear,
        loss_function=mean_squared_error,
        learning_rate=0.001,
        max_iter=300,
        batch_size=32,
        optimizer="adam",
        l2_lambda=0.0001,
        verbose=True
    )
    
    # Prepare data for the custom neural network
    # Convert y_train to proper shape for the custom network
    y_train_reshaped = y_train.reshape(-1, 1)
    
    # Fit the custom neural network
    losses, _ = custom_nn.fit(X_train, y_train_reshaped)
    
    training_time = time.time() - start_time
    print(f"Custom NN Training Time: {training_time:.2f} seconds")
    
    # Evaluate custom neural network
    custom_pred, custom_mse, custom_rmse, custom_r2, custom_pred_time = evaluate_model(
        custom_nn, X_test, y_test, y_scaler, "Custom Neural Network"
    )
    
    return custom_nn, losses, custom_mse, custom_rmse, custom_r2, training_time, custom_pred_time

def test_sklearn_mlp():
    """Test scikit-learn's MLPRegressor."""
    # Load and preprocess data (already sampled)
    X_train, X_test, y_train, y_test, y_scaler = load_and_preprocess_data()
    
    # Train scikit-learn MLPRegressor
    print("\nTraining scikit-learn MLPRegressor...")
    start_time = time.time()
    
    sklearn_mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=300,
        batch_size=32,
        alpha=0.0001,  # L2 regularization
        verbose=True,
        random_state=42
    )
    
    sklearn_mlp.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"scikit-learn MLPRegressor Training Time: {training_time:.2f} seconds")
    
    # Evaluate scikit-learn neural network
    sklearn_pred, sklearn_mse, sklearn_rmse, sklearn_r2, sklearn_pred_time = evaluate_model(
        sklearn_mlp, X_test, y_test, y_scaler, "scikit-learn MLPRegressor"
    )
    
    return sklearn_mlp, sklearn_mse, sklearn_rmse, sklearn_r2, training_time, sklearn_pred_time

def plot_learning_curve(losses):
    """Plot the learning curve of the custom neural network."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Learning Curve - Custom Neural Network')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    # plt.savefig('custom_nn_learning_curve.png')
    plt.close()

def plot_predictions(custom_pred, sklearn_pred, y_test, y_scaler):
    """Plot predictions vs actual values for both models."""
    # Convert scaled test values back to original scale
    y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Sample a subset of points to make the plot clearer
    sample_size = min(200, len(y_test_original))
    indices = np.random.choice(len(y_test_original), sample_size, replace=False)
    
    y_test_sample = y_test_original[indices]
    custom_pred_sample = custom_pred[indices]
    sklearn_pred_sample = sklearn_pred[indices]
    
    # Sort by actual values for better visualization
    sort_idx = np.argsort(y_test_sample)
    y_test_sample = y_test_sample[sort_idx]
    custom_pred_sample = custom_pred_sample[sort_idx]
    sklearn_pred_sample = sklearn_pred_sample[sort_idx]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    plt.scatter(range(len(y_test_sample)), y_test_sample, color='blue', label='Actual', alpha=0.7)
    plt.scatter(range(len(y_test_sample)), custom_pred_sample, color='red', label='Custom NN', alpha=0.5)
    plt.scatter(range(len(y_test_sample)), sklearn_pred_sample, color='green', label='scikit-learn MLPRegressor', alpha=0.5)
    
    plt.title('Predictions vs Actual Values (California Housing)')
    plt.xlabel('Sample Index (sorted by actual value)')
    plt.ylabel('Housing Price')
    plt.legend()
    plt.grid(True)
    # plt.savefig('model_comparison_predictions.png')
    plt.close()
    
    # Plot residuals
    plt.figure(figsize=(12, 8))
    
    custom_residuals = custom_pred_sample - y_test_sample
    sklearn_residuals = sklearn_pred_sample - y_test_sample
    
    plt.scatter(y_test_sample, custom_residuals, color='red', label='Custom NN Residuals', alpha=0.5)
    plt.scatter(y_test_sample, sklearn_residuals, color='green', label='scikit-learn Residuals', alpha=0.5)
    
    plt.axhline(y=0, color='blue', linestyle='-')
    plt.title('Residual Plot (California Housing)')
    plt.xlabel('Actual Value')
    plt.ylabel('Residual (Predicted - Actual)')
    plt.legend()
    plt.grid(True)
    # plt.savefig('model_comparison_residuals.png')
    plt.close()
    
    # Create correlation plot
    plt.figure(figsize=(10, 10))
    
    # Custom NN correlation plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_test_sample, custom_pred_sample, alpha=0.5)
    plt.plot([y_test_sample.min(), y_test_sample.max()], [y_test_sample.min(), y_test_sample.max()], 'k--')
    plt.title('Custom NN: Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # scikit-learn correlation plot
    plt.subplot(2, 2, 2)
    plt.scatter(y_test_sample, sklearn_pred_sample, alpha=0.5)
    plt.plot([y_test_sample.min(), y_test_sample.max()], [y_test_sample.min(), y_test_sample.max()], 'k--')
    plt.title('scikit-learn: Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    # plt.savefig('correlation_plots.png')
    plt.close()

def plot_comparison_metrics(custom_metrics, sklearn_metrics):
    """Plot performance comparison metrics."""
    metrics_names = ['MSE', 'RMSE', 'R²', 'Training Time (s)', 'Prediction Time (s)']
    custom_values = custom_metrics
    sklearn_values = sklearn_metrics
    
    fig, axs = plt.subplots(len(metrics_names), 1, figsize=(10, 15))
    
    for i, (metric, ax) in enumerate(zip(metrics_names, axs)):
        ax.bar(['Custom NN', 'scikit-learn MLP'], [custom_values[i], sklearn_values[i]], color=['red', 'green'])
        ax.set_title(f'{metric} Comparison')
        ax.grid(axis='y')
        
        # Add values on top of bars
        for j, v in enumerate([custom_values[i], sklearn_values[i]]):
            ax.text(j, v, f'{v:.4f}', ha='center', va='bottom')
            
        # For R² score, set y-limits to make differences more visible
        if metric == 'R²':
            min_val = min(custom_values[i], sklearn_values[i]) * 0.95
            ax.set_ylim([min_val, 1.0])
    
    plt.tight_layout()
    # plt.savefig('performance_metrics_comparison.png')
    plt.close()

def main():
    """Main function to run all tests and comparisons."""
    print("=" * 50)
    print("NEURAL NETWORK COMPARISON - CALIFORNIA HOUSING DATASET (SAMPLED)")
    print("=" * 50)
    
    # Define sample size for the entire test
    sample_size = 2000  # Adjust based on your computational resources
    
    # Load data once to get y_scaler for later use
    _, _, _, _, y_scaler = load_and_preprocess_data(sample_size=sample_size)
    
    # Test custom neural network
    custom_nn, losses, custom_mse, custom_rmse, custom_r2, custom_training_time, custom_pred_time = test_custom_nn()
    
    # Test scikit-learn MLPRegressor
    sklearn_mlp, sklearn_mse, sklearn_rmse, sklearn_r2, sklearn_training_time, sklearn_pred_time = test_sklearn_mlp()
    
    # Get predictions for plotting (using the same sample size)
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(sample_size=sample_size)
    
    # Get predictions
    custom_pred = np.array([custom_nn.predict(X_test[i].reshape(1, -1)).flatten() for i in range(X_test.shape[0])])
    sklearn_pred = sklearn_mlp.predict(X_test)
    
    # If predictions are scaled, inverse transform them
    custom_pred_original = y_scaler.inverse_transform(custom_pred.reshape(-1, 1)).flatten()
    sklearn_pred_original = y_scaler.inverse_transform(sklearn_pred.reshape(-1, 1)).flatten()
    
    # Plot learning curve
    plot_learning_curve(losses)
    
    # Plot predictions
    plot_predictions(custom_pred_original, sklearn_pred_original, y_test, y_scaler)
    
    # Plot comparison metrics
    custom_metrics = [custom_mse, custom_rmse, custom_r2, custom_training_time, custom_pred_time]
    sklearn_metrics = [sklearn_mse, sklearn_rmse, sklearn_r2, sklearn_training_time, sklearn_pred_time]
    plot_comparison_metrics(custom_metrics, sklearn_metrics)
    
    # Print summary
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<20} {'Custom NN':<15} {'scikit-learn MLP':<15} {'Difference':<15}")
    print("-" * 65)
    print(f"{'MSE':<20} {custom_mse:<15.4f} {sklearn_mse:<15.4f} {abs(custom_mse - sklearn_mse):<15.4f}")
    print(f"{'RMSE':<20} {custom_rmse:<15.4f} {sklearn_rmse:<15.4f} {abs(custom_rmse - sklearn_rmse):<15.4f}")
    print(f"{'R² Score':<20} {custom_r2:<15.4f} {sklearn_r2:<15.4f} {abs(custom_r2 - sklearn_r2):<15.4f}")
    print(f"{'Training Time (s)':<20} {custom_training_time:<15.2f} {sklearn_training_time:<15.2f} {abs(custom_training_time - sklearn_training_time):<15.2f}")
    print(f"{'Prediction Time (s)':<20} {custom_pred_time:<15.4f} {sklearn_pred_time:<15.4f} {abs(custom_pred_time - sklearn_pred_time):<15.4f}")
    
    # Print conclusion
    print("\n" + "=" * 50)
    print("CONCLUSION")
    print("=" * 50)
    
    better_mse = "Custom NN" if custom_mse < sklearn_mse else "scikit-learn MLP"
    better_r2 = "Custom NN" if custom_r2 > sklearn_r2 else "scikit-learn MLP"
    faster_training = "Custom NN" if custom_training_time < sklearn_training_time else "scikit-learn MLP"
    faster_prediction = "Custom NN" if custom_pred_time < sklearn_pred_time else "scikit-learn MLP"
    
    print(f"Better MSE: {better_mse}")
    print(f"Better R² Score: {better_r2}")
    print(f"Faster Training: {faster_training}")
    print(f"Faster Prediction: {faster_prediction}")
    
    # Overall assessment
    custom_wins = sum([
        1 if custom_mse < sklearn_mse else 0,
        1 if custom_r2 > sklearn_r2 else 0,
        1 if custom_training_time < sklearn_training_time else 0,
        1 if custom_pred_time < sklearn_pred_time else 0
    ])
    
    sklearn_wins = 4 - custom_wins
    
    if custom_wins > sklearn_wins:
        print("\nOverall, the Custom Neural Network implementation performs better on this task.")
    elif sklearn_wins > custom_wins:
        print("\nOverall, the scikit-learn MLPRegressor performs better on this task.")
    else:
        print("\nBoth implementations perform similarly, with each having different strengths.")
    
    # Additional observations
    print("\nAdditional Observations:")
    
    # Compare MSE difference
    mse_diff_percent = abs(custom_mse - sklearn_mse) / min(custom_mse, sklearn_mse) * 100
    if mse_diff_percent < 5:
        print("- The prediction accuracy of both models is very similar (less than 5% difference in MSE).")
    else:
        print(f"- The {better_mse} achieves {mse_diff_percent:.1f}% better MSE than the other model.")
    
    # Compare training speed
    training_diff_percent = abs(custom_training_time - sklearn_training_time) / min(custom_training_time, sklearn_training_time) * 100
    if training_diff_percent > 20:
        print(f"- The {faster_training} is significantly faster in training (by {training_diff_percent:.1f}%).")
    
    # Final recommendation
    print("\nRecommendation:")
    if custom_r2 > sklearn_r2:
        print("The custom neural network implementation is recommended for this task due to its better predictive performance.")
    elif sklearn_r2 > custom_r2 and (sklearn_training_time < custom_training_time):
        print("The scikit-learn MLPRegressor is recommended for this task due to its combination of good performance and faster training.")
    else:
        print("Both implementations are viable options, with the choice depending on specific priorities (speed vs. accuracy).")

if __name__ == "__main__":
    main()