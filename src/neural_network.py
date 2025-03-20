import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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
def categorical_cross_entropy_derivative(y_true, logits):
    probs = softmax(logits)  # Compute softmax probabilities
    return probs - y_true

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
    return (2 / (np.exp(x) - np.exp(-x))) ** 2


#Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Derivative of Softmax
def softmax_derivative(x):
    batch_size, num_classes = x.shape
    jacobian = np.zeros((batch_size, num_classes, num_classes))

    for i in range(batch_size):
        s = x[i].reshape(-1, 1)  # Convert to column vector
        jacobian[i] = np.diagflat(s) - np.dot(s, s.T)

    return jacobian


###################
# Neural Network
###################

class NeuralNetwork:
    def __init__(self, layer_sizes, hidden_layer_activation_functions=[], output_layer_activation_function=sigmoid, loss_function=binary_cross_entropy, learning_rate=0.01, max_iter=1000, batch_size=64, verbose=False):
        self.layer_sizes = layer_sizes
        self.hidden_layer_activation_functions = hidden_layer_activation_functions
        self.output_layer_activation_function = output_layer_activation_function
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.max_iter = max_iter # or max_epoch
        self.batch_size = batch_size
        self.verbose = verbose

        self.hidden_layer_activation_functions = [None] + hidden_layer_activation_functions + [None]
        self.hidden_layer_activation_derivatives = []
        self.output_layer_activation_derivative = None
        self.loss_derivative = None

        self.weights = [None]
        self.biases = [None]

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
            else:
                print("Hidden layer activation function not yet implemented!")
                exit(0)

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

        K = len(self.layer_sizes)
        # Normal random initialization
        for i in range(1, K):
            # W_i is of size l_{l-1} x l_{i}
            self.weights.append(np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]))
            # b_i is of size 1 x l_i
            self.biases.append(np.random.randn(1, self.layer_sizes[i]) * 0.01)

        # TODO: Zero initialization
        # TODO: Uniform random initialization
        # TODO: Bonus: Xavier initialization
        # TODO: Bonus: He initialization

        # TODO: Manual seeding for reproducibility in all initialiations according to spec.

    
    def forward_propagation(self, X):
        K = len(self.layer_sizes)

        preactivations = []
        activations = []

        # 1. Input layer: A_0 = Z_0 = X
        preactivations.append(X)
        activations.append(X)

        # 2. Hidden layers:
        # Preactivation is defined by Z_i = A_{i-1}W_i + B_i, i = 1, 2, ..., k-1
        # Activation is defined by A_i = f_i(Z_i), i = 1, 2, ..., k-2, where f_i is the activation function in the ith hidden layer applied elementwise.
        for i in range(1, K-1): # Calculates Z_i & A_i for i = 1, ..., k-2
            Z_i = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            f_i = self.hidden_layer_activation_functions[i]
            A_i = f_i(Z_i)
            preactivations.append(Z_i)
            activations.append(A_i)
        

        # 3. Output layer: # i = k-1
        # The activation is a special activation function since it is the output layer activation function so it depends on the task being done (for instance, sigmoid for binary classification)
        preactivations.append(np.dot(activations[-1], self.weights[K-1]) + self.biases[K-1])
        activations.append(self.output_layer_activation_function(preactivations[-1]))
        
        # assert (len(preactivations) == len(activations) + 1)
        for i in range(1,K):
            assert activations[i].shape == (1, self.layer_sizes[i]) or activations[i].shape == (self.layer_sizes[i],)

        # print("Berhasil # 1")
        # exit(0)

        return (preactivations, activations)
    

    def backward_propagation(self, preactivations, activations, X, y):
        K = len(self.layer_sizes)

        # 1. Backward pass 1: Calculating dl/dz_i for i = k-1, ..., 1
        dldz_i = []
        # a. Base case, i = k-1
        # Calculates Q1 = f'_{k-1}(z_{k-1}). Note that f_{k-1} is the output layer activation function
        Q1 = self.output_layer_activation_derivative(preactivations[K-1])
        assert Q1.shape == (1, self.layer_sizes[K-1]) or Q1.shape == (self.layer_sizes[K-1],)
        
        # Calculates Q2: dl/da_{k-1}
        Q2 = self.loss_derivative(y, activations[-1])
        assert Q2.shape == (1, self.layer_sizes[K-1]) or Q2.shape == (self.layer_sizes[K-1],)
        
        # The final result is Q1 ⊙ Q2 where ⊙ is component-wise multiplication
        assert (Q1 * Q2).shape == (1, self.layer_sizes[K-1]) or (Q1 * Q2).shape == (self.layer_sizes[K-1],)
        dldz_i.append(Q1 * Q2)

        # print("Berhasil # 2")
        # exit(0)
        
        
        # b. Inductive case, i = k-2, .., 1
        for i in range(K-2, 0, -1):
            # Calculates Q1 = f'_i(z_i)
            Q1 = self.hidden_layer_activation_derivatives[i](preactivations[i])
            assert Q1.shape == (1, self.layer_sizes[i]) or Q1.shape == (self.layer_sizes[i],)
            # print("Berhasil # 3")
            # exit(0)

            # Calculates Q2 = dl/dz_{i+1} (W_{i+1})^T
            Q2 = np.dot(dldz_i[-1], self.weights[i+1].T)
            assert Q2.shape == (1, self.layer_sizes[i]) or Q2.shape == (self.layer_sizes[i],)
            # print("Berhasil # 4")
            # exit(0)

            # The final result is Q1 ⊙ Q2 where ⊙ is component-wise multiplication
            assert (Q1 * Q2).shape == (1, self.layer_sizes[i]) or (Q1 * Q2).shape == (self.layer_sizes[i],)
            # print("Berhasil # 5")
            # exit(0) 
            dldz_i.append(Q1 * Q2)

        dldz_i.append(None)
        dldz_i.reverse()

        assert len(dldz_i) == K

        # 2. Backward pass 2: Calculating dl/db_i and dl/dW_i for i = 1, 2, ..., k-1

        dldbi = [None]
        dldWi = [None]

        for i in range(1, K):
            # dl/dbi = dl/dzi
            dldbi.append(dldz_i[i])
            # print("i = ", i)
            assert np.dot(activations[i-1].T.reshape(-1, 1), dldz_i[i]).shape == (self.layer_sizes[i-1], self.layer_sizes[i])
            # print("Berhasil # 6")
            # exit(0)

            # dl/dwi = A_{i-1}^T dl/dz_i
            dldWi.append(np.dot(activations[i-1].T.reshape(-1, 1), dldz_i[i]))

        assert len(self.weights) == len(dldWi)
        assert len(self.biases) == len(dldbi)

        return (dldWi, dldbi)
        

    def calculate_updated_weights_and_biases(self, X, y):
        preactivations, activations = self.forward_propagation(X)
        return self.backward_propagation(preactivations, activations, X, y)
    
    def fit(self, X, y):
        n_input = X.shape[0]
        n_batches = max(n_input // self.batch_size, 1)
        
        # For every iteration/epoch,
        for i in range(self.max_iter):
            print(i, "th epoch/iteration")
            # print("Weights", self.weights)
            # print("Biases", self.biases)

            # Shuffle
            indices = np.random.permutation(n_input)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Choose a random batch
            for b in range(n_batches):
                start_idx = b * self.batch_size
                end_idx = min((b + 1) * self.batch_size, n_input)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Sum gradients for each input in the batch
                weights_update = [np.zeros_like(w) if w is not None else None for w in self.weights]
                biases_update = [np.zeros_like(b) if b is not None else None for b in self.biases]

                assert len(X_batch) == len(y_batch) and len(X_batch) == self.batch_size

                for X_inp, y_inp in zip(X_batch, y_batch):
                    current_weights_update, current_biases_update = self.calculate_updated_weights_and_biases(X_inp, y_inp)
                    weights_update = [w1 + w2 if w1 is not None else None for w1, w2 in zip(weights_update, current_weights_update)]
                    biases_update = [w1 + w2 if w1 is not None else None for w1, w2 in zip(biases_update, current_biases_update)]

                # Update weights and biases
                self.weights = [w1 - self.learning_rate * w2 / self.batch_size if w1 is not None else None for w1, w2 in zip(self.weights, weights_update)]
                self.biases = [w1 - self.learning_rate * w2 / self.batch_size if w1 is not None else None for w1, w2 in zip(self.biases, biases_update)]
            
    
    def predict(self, X):
        # Check if X is a single input or multiple inputs
        if X.shape[0] == 1:
            # Single input case - just process directly
            _, activations = self.forward_propagation(X)
            y_pred = activations[-1] >= 0.5
            return y_pred.astype(int)
        else:
            # Process each row individually and stack results
            predictions = []
            for i in range(X.shape[0]):
                single_input = X[i:i+1]  # Keep it as a 2D array with shape (1, features)
                _, activations = self.forward_propagation(single_input)
                pred = activations[-1] >= 0.5

                predictions.append(pred.astype(int))
            
            # Stack all predictions together
            return np.vstack(predictions)
    
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)