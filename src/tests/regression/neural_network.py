import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import networkx as nx


##########################################
# Loss functions and their derivatives:
##########################################

# Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Derivative of Mean Squared Error
def mean_squared_error_derivative(y_true, y_pred):
    return (2 / y_true.shape[0]) * (y_pred - y_true)

# Binary Cross-Entropy
def binary_cross_entropy(y_true, y_pred):
    # Adding a small value to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Binary Cross-Entropy Derivative
def binary_cross_entropy_derivative(y_true, y_pred):
    # Adding a small value to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.size

# Categorical Cross-Entropy
def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-12  # Avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Derivative of Categorical Cross-Entropy
def categorical_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return (y_pred - y_true) / y_true.shape[0]

################################################
# Activation functions and their derivatives:
################################################

# Linear function
def linear(x):
    return x

# Derivative of Linear
def linear_derivative(x):
    return np.ones_like(x)


# ReLU function
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Sigmoid function
def sigmoid(x):
    # Clip values to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# Tanh function
def tanh(x):
    return np.tanh(x)

# Derivative of Tanh
def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

# Softmax function
def softmax(x):
    # Always shift values for numerical stability
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    # Clip to avoid overflow/underflow
    shifted_x = np.clip(shifted_x, -500, 500)
    # Calculate softmax
    exp_x = np.exp(shifted_x)
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-12)

# Derivative of Softmax (Jacobian matrix)
def softmax_derivative(x):
    # For numerical stability
    x = np.clip(x, -500, 500)
    
    # Get the softmax probabilities
    s = softmax(x)
    
    # Handle 1D input case (single sample vector)
    if len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1):
        # Flatten if it's a single row matrix
        if len(x.shape) == 2:
            s = s.flatten()
            
        n_classes = len(s)
        # Create the Jacobian matrix
        jacobian = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                if i == j:
                    jacobian[i, j] = s[i] * (1 - s[i])
                else:
                    jacobian[i, j] = -s[i] * s[j]
        
        return jacobian
    
    # For multi-sample case, we handle only the first sample since our neural network implementation processes serially,
    # basically intead of batch_size x a x b, it's a x b.
    print("Warning: softmax_derivative received multiple samples. Only using the first sample because I don't know math.")
    s_first = s[0]
    n_classes = len(s_first)
    
    jacobian = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                jacobian[i, j] = s_first[i] * (1 - s_first[i])
            else:
                jacobian[i, j] = -s_first[i] * s_first[j]
    
    return jacobian

class NeuralNetwork:
    def __init__(self, layer_sizes, initialization_method=None, hidden_layer_activation_functions=[], 
                 output_layer_activation_function=None, loss_function=None, learning_rate=0.01, 
                 max_iter=1000, batch_size=64, optimizer="sgd", beta1=0.9, beta2=0.999, epsilon=1e-8, 
                 l1_lambda=0.0, l2_lambda=0.0, verbose=False, seed=42):
        # Add random seed for reproducibility
        self.seed = seed
        np.random.seed(self.seed)
        
        self.layer_sizes = layer_sizes
        self.hidden_layer_activation_functions = hidden_layer_activation_functions
        self.output_layer_activation_function = output_layer_activation_function
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.initialization_method = initialization_method
        self.verbose = verbose
        self.losses = []
        self.val_losses = []
        self.gradients = []
        
        # Regularization parameters
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        # Adam optimizer parameters
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Adam optimizer moment variables
        self.m_weights = None
        self.v_weights = None
        self.m_biases = None
        self.v_biases = None
        self.t = 0
        
        # Set up activation functions and derivatives
        self.hidden_layer_activation_functions = [None] + hidden_layer_activation_functions + [None]
        self.hidden_layer_activation_derivatives = []
        self.output_layer_activation_derivative = None
        self.loss_derivative = None
        
        # Initialize weights and biases
        self.weights = [None]
        self.biases = [None]
        
        # Initialize activation derivatives based on activation functions
        for func in self.hidden_layer_activation_functions:
            if func is None:
                self.hidden_layer_activation_derivatives.append(None)
                continue
                
            if func == linear:
                self.hidden_layer_activation_derivatives.append(linear_derivative)
            elif func == relu:
                self.hidden_layer_activation_derivatives.append(relu_derivative)
            elif func == sigmoid:
                self.hidden_layer_activation_derivatives.append(sigmoid_derivative)
            elif func == tanh:
                self.hidden_layer_activation_derivatives.append(tanh_derivative)
            elif func == softmax:
                self.hidden_layer_activation_derivatives.append(softmax_derivative)
            else:
                print("Hidden layer activation function not yet implemented!")
                exit(0)
                
        # Set output layer activation derivative
        if self.output_layer_activation_function == linear:
            self.output_layer_activation_derivative = linear_derivative
        elif self.output_layer_activation_function == relu:
            self.output_layer_activation_derivative = relu_derivative
        elif self.output_layer_activation_function == sigmoid:
            self.output_layer_activation_derivative = sigmoid_derivative
        elif self.output_layer_activation_function == tanh:
            self.output_layer_activation_derivative = tanh_derivative
        elif self.output_layer_activation_function == softmax:
            self.output_layer_activation_derivative = softmax_derivative
        else:
            print("Output layer activation layer not yet implemented!")
            exit(0)
            
        # Set loss derivative
        if self.loss_function == mean_squared_error:
            self.loss_derivative = mean_squared_error_derivative
        elif self.loss_function == binary_cross_entropy:
            self.loss_derivative = binary_cross_entropy_derivative
        elif self.loss_function == categorical_cross_entropy:
            self.loss_derivative = categorical_cross_entropy_derivative
        else:
            print("Loss function not yet implemented!")
            exit(0)
            
        assert self.loss_derivative is not None
        assert self.output_layer_activation_derivative is not None
        assert len(self.hidden_layer_activation_derivatives) == len(self.hidden_layer_activation_functions)
        
        # Determine initialization method if not provided
        if self.initialization_method is None:
            if any(func == relu for func in self.hidden_layer_activation_functions):
                self.initialization_method = "he"
            elif any(func in [sigmoid, tanh] for func in self.hidden_layer_activation_functions):
                self.initialization_method = "xavier"
            else:
                self.initialization_method = "xavier"
                
        # Initialize weights and biases
        K = len(self.layer_sizes)
        
        for i in range(1, K):
            if self.initialization_method == "zero":
                self.weights.append(np.zeros((self.layer_sizes[i-1], self.layer_sizes[i])))
                self.biases.append(np.zeros((1, self.layer_sizes[i])))
            
            elif self.initialization_method == "random":
                self.weights.append(np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * 1)
                self.biases.append(np.random.randn(1, self.layer_sizes[i]) * 0.001)
            elif self.initialization_method == "random_uniform":
                self.weights.append(np.random.rand(self.layer_sizes[i-1], self.layer_sizes[i]) * 1)
                self.biases.append(np.random.rand(1, self.layer_sizes[i]) * 0.001)
            elif self.initialization_method == "xavier":
                limit = np.sqrt(6 / (self.layer_sizes[i-1] + self.layer_sizes[i]))
                self.weights.append(np.random.uniform(-limit, limit, (self.layer_sizes[i-1], self.layer_sizes[i])))
                self.biases.append(np.zeros((1, self.layer_sizes[i])))
            
            elif self.initialization_method == "he":
                std = np.sqrt(2 / self.layer_sizes[i-1])
                self.weights.append(np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * std)
                self.biases.append(np.zeros((1, self.layer_sizes[i])))
            
            else:
                raise ValueError(f"Unknown initialization method: {self.initialization_method}")
                
        # Initialize Adam optimizer variables if using Adam
        if self.optimizer == "adam":
            self.initialize_adam_variables()
    
    def initialize_adam_variables(self):
        # Initialize the moment variables for Adam optimizer
        self.m_weights = [None]
        self.v_weights = [None]
        self.m_biases = [None]
        self.v_biases = [None]
        
        for i in range(1, len(self.layer_sizes)):
            self.m_weights.append(np.zeros_like(self.weights[i]))
            self.v_weights.append(np.zeros_like(self.weights[i]))
            self.m_biases.append(np.zeros_like(self.biases[i]))
            self.v_biases.append(np.zeros_like(self.biases[i]))
        
        self.t = 0

    def forward_propagation(self, X):
        K = len(self.layer_sizes)

        preactivations = []
        activations = []

        # Input layer
        preactivations.append(X)
        activations.append(X)

        # Hidden layers
        for i in range(1, K-1):
            Z_i = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            f_i = self.hidden_layer_activation_functions[i]
            A_i = f_i(Z_i)
            preactivations.append(Z_i)
            activations.append(A_i)

        # Output layer
        preactivations.append(np.dot(activations[-1], self.weights[K-1]) + self.biases[K-1])
        activations.append(self.output_layer_activation_function(preactivations[-1]))


        for i in range(1, K):
            if len(activations[i].shape) == 1:
                activations[i] = activations[i].reshape(1, -1)

        return (preactivations, activations)
    
    def calculate_total_loss(self, y, y_pred):
        # Calculate the original loss function
        original_loss = self.loss_function(y, y_pred)
        
        # Calculate L1 regularization term
        l1_reg_term = 0.0
        if self.l1_lambda > 0:
            for i in range(1, len(self.weights)):
                if self.weights[i] is not None:
                    l1_reg_term += np.sum(np.abs(self.weights[i]))
        
        # Calculate L2 regularization term
        l2_reg_term = 0.0
        if self.l2_lambda > 0:
            for i in range(1, len(self.weights)):
                if self.weights[i] is not None:
                    l2_reg_term += np.sum(np.square(self.weights[i]))
        
        # Return the combined loss
        return original_loss + self.l1_lambda * l1_reg_term + 0.5 * self.l2_lambda * l2_reg_term

    def backward_propagation(self, preactivations, activations, X, y):
        K = len(self.layer_sizes)
        batch_size = X.shape[0] if len(X.shape) > 1 else 1

        # Calculate dl/dz_i for output and hidden layers
        dldz_i = []

        # Special case for softmax output activation layer + categorical cross entropy loss function
        # Note that this is not necessary since Q2 Q1 still gives the correct answer with Q2 being the jacobian matrix, however, this severely improves
        # the performance and most importantly the numerical stability due to large intermediate numbers when calculating Q2 Q1.
        # My previous testing indicates that doing Q2 Q1 for this case flutuates the model accuracy due to numerical instability 
        if self.output_layer_activation_function == softmax and self.loss_function == categorical_cross_entropy:
            # Combined derivative simplifies to (predictions - targets)
            dldz_K_minus_1 = activations[-1] - y
            dldz_i.append(dldz_K_minus_1)
        else:
            # For other activation/loss combinations
            # Handle batch processing by applying the activation derivative to each example
            dZ = np.zeros_like(preactivations[K-1])
            
            # Apply activation derivative to each example in batch
            for b in range(batch_size):
                sample_preact = preactivations[K-1][b:b+1] if batch_size > 1 else preactivations[K-1]
                sample_act = activations[-1][b:b+1] if batch_size > 1 else activations[-1]
                sample_y = y[b:b+1] if batch_size > 1 else y
                
                Q1 = self.output_layer_activation_derivative(sample_preact)
                
                if self.output_layer_activation_function != softmax:
                    if len(Q1.shape) == 1 or (len(Q1.shape) == 2 and Q1.shape[0] == 1):
                        Q1 = np.diag(Q1.flatten())
                        
                # Calculate loss derivative
                Q2 = self.loss_derivative(sample_y, sample_act)
                
                # Apply to current example
                dZ[b] = np.dot(Q2, Q1).flatten() if batch_size > 1 else np.dot(Q2, Q1)
                
            dldz_i.append(dZ)

        # Inductive case for hidden layers
        for i in range(K-2, 0, -1):
            # Initialize gradient for this layer
            dZ = np.zeros_like(preactivations[i])
            
            # Get upstream gradient
            dA = dldz_i[-1]
            
            # Handle gradient for each example in batch
            for b in range(batch_size):
                # Get sample activation derivative
                sample_preact = preactivations[i][b:b+1] if batch_size > 1 else preactivations[i]
                Q1 = self.hidden_layer_activation_derivatives[i](sample_preact)
                
                if len(Q1.shape) == 1 or (len(Q1.shape) == 2 and Q1.shape[0] == 1):
                    Q1 = np.diag(Q1.flatten())
                
                # Get sample upstream gradient
                sample_dA = dA[b:b+1] if batch_size > 1 else dA
                
                # Calculate gradient for this example
                Q2 = np.dot(sample_dA, self.weights[i+1].T)
                dZ[b] = np.dot(Q2, Q1).flatten() if batch_size > 1 else np.dot(Q2, Q1)
                
            dldz_i.append(dZ)

        dldz_i.append(None)
        dldz_i.reverse()

        assert len(dldz_i) == K

        # Calculate dl/db_i and dl/dW_i
        dldbi = [None]
        dldWi = [None]

        for i in range(1, K):
            # dl/dbi = sum of dl/dzi across batch
            dldbi.append(np.sum(dldz_i[i], axis=0, keepdims=True))
            
            # dl/dwi = A_{i-1}^T dl/dz_i + regularization terms
            A_prev = activations[i-1]
            dZ = dldz_i[i]
            
            if len(A_prev.shape) == 1:
                A_prev = A_prev.reshape(1, -1)
            if len(dZ.shape) == 1:
                dZ = dZ.reshape(1, -1)
                
            # Calculate gradients for all examples and sum
            base_gradient = np.zeros_like(self.weights[i])
            for b in range(batch_size):
                a_prev_sample = A_prev[b:b+1].T if batch_size > 1 else A_prev.T
                dz_sample = dZ[b:b+1] if batch_size > 1 else dZ
                base_gradient += np.dot(a_prev_sample, dz_sample)
                
            # Add L1 regularization gradient
            l1_gradient = 0
            if self.l1_lambda > 0:
                # The gradient of L1 is sign(w)
                l1_gradient = self.l1_lambda * np.sign(self.weights[i])
            
            # Add L2 regularization gradient
            l2_gradient = 0
            if self.l2_lambda > 0:
                # The gradient of L2 is w
                l2_gradient = self.l2_lambda * self.weights[i]
            
            # Combine gradients
            dldWi.append(base_gradient + l1_gradient + l2_gradient)


        assert len(self.weights) == len(dldWi)
        assert len(self.biases) == len(dldbi)
        
        return (dldWi, dldbi)



    def apply_adam_update(self, weights_update, biases_update, batch_size):
        # Apply Adam optimizer update to weights and biases
        self.t += 1
        
        new_weights = [None]
        new_biases = [None]
        
        for i in range(1, len(self.weights)):
            if self.weights[i] is None:
                new_weights.append(None)
                new_biases.append(None)
                continue
                
            # Normalize gradients by batch size
            grad_w = weights_update[i] / batch_size
            grad_b = biases_update[i] / batch_size
            
            # Update biased first moment estimates
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * grad_w
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * grad_b
            
            # Update biased second moment estimates
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (grad_w**2)
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (grad_b**2)
            
            # Compute bias-corrected first moment estimates
            m_weights_corrected = self.m_weights[i] / (1 - self.beta1 ** self.t)
            m_biases_corrected = self.m_biases[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimates
            v_weights_corrected = self.v_weights[i] / (1 - self.beta2 ** self.t)
            v_biases_corrected = self.v_biases[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            new_weights.append(self.weights[i] - self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon))
            new_biases.append(self.biases[i] - self.learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + self.epsilon))
            
        return new_weights, new_biases

    def fit(self, X, y, X_val=None, y_val=None):
        n_input = X.shape[0]
        n_batches = max(n_input // self.batch_size, 1)
        
        self.losses = []
        self.val_losses = []
        final_gradients = [None] * len(self.weights)
        
        # For every epoch/iteration
        for i in range(self.max_iter):
            epoch_loss = 0.0
            current_epoch_gradients = []
            
            # Learning rate schedule - only for SGD, not for Adam
            if self.optimizer == "sgd":
                current_learning_rate = self.learning_rate / (1 + 0.0001 * i)
            else:
                current_learning_rate = self.learning_rate
                
            # Shuffle data with consistent seed for reproducibility
            # Use a different seed for each epoch but derived from the base seed
            np.random.seed(self.seed + i)
            indices = np.random.permutation(n_input)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process each batch
            for b in range(n_batches):
                start_idx = b * self.batch_size
                end_idx = min((b + 1) * self.batch_size, n_input)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass with entire batch
                preactivations, activations = self.forward_propagation(X_batch)
                
                # Calculate batch loss with L1 and L2 regularization
                batch_loss = 0.0
                for j in range(len(X_batch)):
                    batch_loss += self.calculate_total_loss(y_batch[j], activations[-1][j])
                batch_loss /= len(X_batch)
                epoch_loss += batch_loss
                
                # Backward pass with the entire batch
                weights_update, biases_update = self.backward_propagation(
                    preactivations, activations, X_batch, y_batch)
                
                # Add gradient clipping to prevent explosion
                max_grad_norm = 5
                weights_update = [np.clip(w, -max_grad_norm, max_grad_norm) if w is not None else None for w in weights_update]
                biases_update = [np.clip(b, -max_grad_norm, max_grad_norm) if b is not None else None for b in biases_update]
                
                # Update weights and biases
                batch_size = len(X_batch)
                
                if self.optimizer == "adam":
                    # Use Adam optimizer
                    self.weights, self.biases = self.apply_adam_update(weights_update, biases_update, batch_size)
                else:
                    # Use SGD
                    self.weights = [w1 - current_learning_rate * w2 / batch_size if w1 is not None else None for w1, w2 in zip(self.weights, weights_update)]
                    self.biases = [b1 - current_learning_rate * b2 / batch_size if b1 is not None else None for b1, b2 in zip(self.biases, biases_update)]
                
                # Store final gradients (after last iteration)
                if i == self.max_iter - 1:
                    final_gradients = weights_update
                    self.gradients = final_gradients

            # Average loss for this epoch
            epoch_loss /= n_batches
            self.losses.append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_loss = self.calculate_validation_loss(X_val, y_val)
                self.val_losses.append(val_loss)
            
            # Print progress
            if self.verbose and i % 100 == 0:
                print(f"Epoch {i}, Loss: {epoch_loss:.6f}, Learning rate: {current_learning_rate:.6f}")
                if X_val is not None and y_val is not None:
                    print(f"Epoch {i}, Validation Loss: {val_loss:.6f}")
                
        return self.losses, final_gradients

    
    def calculate_validation_loss(self, X_val, y_val):
        batch_size = 32
        n_samples = X_val.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        val_loss = 0.0
        
        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, n_samples)
            
            X_batch = X_val[start_idx:end_idx]
            y_batch = y_val[start_idx:end_idx]
            
            _, activations = self.forward_propagation(X_batch)
            
            batch_loss = 0.0
            for j in range(len(X_batch)):
                batch_loss += self.calculate_total_loss(y_batch[j], activations[-1][j])
            
            val_loss += batch_loss
        
        return val_loss / n_samples

    def predict(self, X):
        _, activations = self.forward_propagation(X)
        return activations[-1]

    def visualize_neural_network(self, max_neurons_to_display=50, save_path=None):
        """
        Visualize the neural network structure with weights and biases.
        
        Parameters:
        ----------
        max_neurons_to_display : int, optional
            Maximum number of neurons to display per layer to prevent overcrowding
        save_path : str, optional
            Path to save the visualization image
        """
        # Create a new figure
        plt.figure(figsize=(15, 10))
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Prepare position layout
        pos = {}
        layer_spacing = 1.0
        neuron_spacing = 0.2
        
        # Compute layer positions and add nodes
        for layer_idx, layer_size in enumerate(self.layer_sizes[1:], start=1):
            # Limit number of displayed neurons
            displayed_neurons = min(layer_size, max_neurons_to_display)
            
            for neuron_idx in range(displayed_neurons):
                # X position based on layer
                x = layer_idx * layer_spacing
                
                # Y position based on neuron index, centered
                y = neuron_idx * neuron_spacing - (displayed_neurons * neuron_spacing / 2)
                
                # Create unique node identifier
                node_id = f'L{layer_idx}_N{neuron_idx}'
                
                # Add node with color representing bias
                bias = self.biases[layer_idx][0, neuron_idx] if self.biases[layer_idx] is not None else 0
                
                G.add_node(node_id, pos=(x, y), bias=bias)
                pos[node_id] = (x, y)
        
        # Add edges with weights
        for layer_idx in range(1, len(self.layer_sizes)-1):
            prev_layer_neurons = min(self.layer_sizes[layer_idx], max_neurons_to_display)
            curr_layer_neurons = min(self.layer_sizes[layer_idx+1], max_neurons_to_display)
            
            for prev_neuron in range(prev_layer_neurons):
                for curr_neuron in range(curr_layer_neurons):
                    # Unique identifiers
                    from_node = f'L{layer_idx}_N{prev_neuron}'
                    to_node = f'L{layer_idx+1}_N{curr_neuron}'
                    
                    # Get weight
                    weight = self.weights[layer_idx+1][prev_neuron, curr_neuron]
                    
                    # Add weighted edge
                    G.add_edge(from_node, to_node, weight=weight)
        
        # Prepare color mappings
        def safe_normalize(values):
            """Safely normalize values to [0, 1] range."""
            if not values:
                return lambda x: 0.5
            min_val, max_val = min(values), max(values)
            return lambda x: 0.5 if min_val == max_val else (x - min_val) / (max_val - min_val)
        
        # Collect bias and weight values for normalization
        bias_values = [G.nodes[node]['bias'] for node in G.nodes()]
        weight_values = [G[u][v]['weight'] for (u,v) in G.edges()]
        
        # Create normalizers
        bias_normalizer = safe_normalize(bias_values)
        weight_normalizer = safe_normalize(weight_values)
        
        # Draw the graph
        plt.title('Neural Network Structure Visualization')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                                node_color=[plt.cm.coolwarm(bias_normalizer(G.nodes[node]['bias'])) 
                                            for node in G.nodes()],
                                node_size=50, 
                                alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                                edge_color=[plt.cm.RdBu_r(weight_normalizer(G[u][v]['weight'])) 
                                            for (u,v) in G.edges()],
                                width=1, 
                                alpha=0.5)
        
        # Finalize plot
        plt.axis('off')
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()