import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import the custom neural network implementation
from neural_network import NeuralNetwork, sigmoid, relu, tanh, mean_squared_error, linear

def load_energy_efficiency_dataset():
    """
    Load the Energy Efficiency dataset
    Source: https://archive.ics.uci.edu/ml/datasets/energy+efficiency
    
    The dataset contains 8 features and 2 targets (heating load and cooling load)
    We'll focus on predicting the heating load as our regression task
    """
    try:
        # Try to load directly from scikit-learn's datasets
        data = fetch_openml(name='energy-efficiency', version=1, as_frame=True)
        X = data.data
        y = data.target.iloc[:, 0]  # Use only the first target (heating load)
    except:
        # If not available, load from a URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        try:
            df = pd.read_excel(url)
        except:
            # If URL doesn't work, create synthetic data that mimics the energy efficiency dataset
            print("Could not access dataset, creating synthetic data with similar properties")
            features = 8
            samples = 768
            np.random.seed(42)
            X = np.random.randn(samples, features)
            # Make y depend on X in a non-linear way
            y = 20 + 3*X[:, 0]**2 - 2*X[:, 1] + 5*np.sin(X[:, 2]) + np.exp(X[:, 3]/10) + 2*X[:, 4] - X[:, 5]**2 + X[:, 6]*X[:, 7]
            y = y + np.random.randn(samples) * 3  # Add some noise
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(features)])
            return X, y
            
        # Column names for the real dataset
        column_names = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
                        'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution',
                        'Heating Load', 'Cooling Load']
        
        if len(df.columns) != len(column_names):
            df.columns = column_names
            
        X = df.iloc[:, :-2]  # All columns except the last two
        y = df.iloc[:, -2]   # Heating Load (second to last column)

    return X, y

def train_custom_nn(X_train, y_train, X_test, y_test):
    """
    Train our custom neural network for regression task
    """
    # Convert pandas to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values.reshape(-1, 1)
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values.reshape(-1, 1)
    
    # Set up the network architecture
    input_size = X_train.shape[1]
    layer_sizes = [input_size, 16, 8, 1]
    
    # Initialize the neural network for regression
    custom_nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        hidden_layer_activation_functions=[relu, relu],
        output_layer_activation_function=linear,  # Linear activation for regression
        loss_function=mean_squared_error,
        learning_rate=0.01,
        max_iter=500,
        batch_size=32,
        optimizer="adam",
        l2_lambda=0.001,  # Slight regularization
        verbose=True
    )
    
    # Train the model
    train_losses = custom_nn.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = np.array([pred.flatten()[0] for pred in custom_nn.predict(X_train)])
    y_pred_test = np.array([pred.flatten()[0] for pred in custom_nn.predict(X_test)])
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Return the results
    return {
        'model': custom_nn,
        'train_losses': train_losses,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_pred_test': y_pred_test
    }

def train_sklearn_mlp(X_train, y_train, X_test, y_test):
    """
    Train scikit-learn's MLPRegressor for comparison
    """
    # Configure MLP to be as similar as possible to our custom NN
    sklearn_mlp = MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization parameter
        batch_size=32,
        learning_rate_init=0.01,
        max_iter=500,
        verbose=True,
        random_state=42
    )
    
    # Train the model
    sklearn_mlp.fit(X_train, y_train.ravel() if isinstance(y_train, np.ndarray) and y_train.ndim > 1 else y_train)
    
    # Make predictions
    y_pred_train = sklearn_mlp.predict(X_train)
    y_pred_test = sklearn_mlp.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Return the results
    return {
        'model': sklearn_mlp,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_pred_test': y_pred_test
    }

def plot_results(custom_nn_results, sklearn_results, X_test, y_test):
    """
    Plot the results comparing both models
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training Loss for Custom NN
    axes[0, 0].plot(custom_nn_results['train_losses'])
    axes[0, 0].set_title('Custom NN Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].grid(True)
    
    # Plot 2: Predicted vs Actual values (Custom NN)
    axes[0, 1].scatter(y_test, custom_nn_results['y_pred_test'], alpha=0.5)
    min_val = min(min(y_test), min(custom_nn_results['y_pred_test']))
    max_val = max(max(y_test), max(custom_nn_results['y_pred_test']))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[0, 1].set_title('Custom NN: Predicted vs Actual')
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].grid(True)
    
    # Plot 3: Predicted vs Actual values (scikit-learn MLP)
    axes[1, 0].scatter(y_test, sklearn_results['y_pred_test'], alpha=0.5)
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1, 0].set_title('Scikit-learn MLP: Predicted vs Actual')
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].grid(True)
    
    # Plot 4: Performance comparison (bar chart)
    metrics = ['Train MSE', 'Test MSE', 'Train R²', 'Test R²']
    custom_values = [custom_nn_results['train_mse'], custom_nn_results['test_mse'], 
                     custom_nn_results['train_r2'], custom_nn_results['test_r2']]
    sklearn_values = [sklearn_results['train_mse'], sklearn_results['test_mse'], 
                     sklearn_results['train_r2'], sklearn_results['test_r2']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, custom_values, width, label='Custom NN')
    axes[1, 1].bar(x + width/2, sklearn_values, width, label='Scikit-learn MLP')
    axes[1, 1].set_title('Performance Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    # plt.savefig('nn_comparison_results.png')
    plt.show()

def main():
    # Linear activation function for regression output
    def linear(x):
        return x
    
    # Load the dataset
    print("Loading Energy Efficiency dataset...")
    X, y = load_energy_efficiency_dataset()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Convert to numpy arrays and reshape if needed
    if isinstance(y_train, pd.Series):
        y_train_array = y_train.values.reshape(-1, 1)
    else:
        y_train_array = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        
    if isinstance(y_test, pd.Series):
        y_test_array = y_test.values.reshape(-1, 1)
    else:
        y_test_array = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
    
    # Train custom neural network
    print("\n=== Training Custom Neural Network ===")
    custom_nn_results = train_custom_nn(X_train, y_train_array, X_test, y_test_array)
    
    # Train scikit-learn MLP
    print("\n=== Training Scikit-learn MLP Regressor ===")
    sklearn_results = train_sklearn_mlp(X_train, y_train, X_test, y_test)
    
    # Print comparison results
    print("\n=== Model Comparison Results ===")
    print(f"{'Metric':<20} {'Custom NN':<15} {'Scikit-learn MLP':<15}")
    print("-" * 50)
    print(f"{'Training MSE':<20} {custom_nn_results['train_mse']:<15.4f} {sklearn_results['train_mse']:<15.4f}")
    print(f"{'Test MSE':<20} {custom_nn_results['test_mse']:<15.4f} {sklearn_results['test_mse']:<15.4f}")
    print(f"{'Training R²':<20} {custom_nn_results['train_r2']:<15.4f} {sklearn_results['train_r2']:<15.4f}")
    print(f"{'Test R²':<20} {custom_nn_results['test_r2']:<15.4f} {sklearn_results['test_r2']:<15.4f}")
    
    # Plot the results
    plot_results(custom_nn_results, sklearn_results, X_test, y_test)
    

if __name__ == "__main__":
    main()