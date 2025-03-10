import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import numpy as np

X, y = make_moons(n_samples=1000, noise=0.2)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Neural Network
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, max_iter=1000, batch_size=64, verbose=False):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter # or max_epoch
        self.batch_size = batch_size
        self.verbose = verbose

        self.weights = []
        self.biases = []

        # Normal random initialization
        for l in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(self.layer_sizes[l], self.layer_sizes[l+1]))
            
            # Biases are usually smaller than weights, so a 0.01 factor is given.
            self.biases.append(np.random.randn(1, self.layer_sizes[l+1]) * 0.01)


        # TODO: Uniform random initialization
        # TODO: Bonus: Xavier initialization
        # TODO: Bonus: He initialization

        # TODO: Manual seeding for reproducibility in all initialiations according to spec.

    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    def relu(self, z):
        return np.maximum(0, z)
    
    def forward_propagation(self, X):
        preactivations = []
        activations = []

        # 1. Input layer: In the first layer or input layer, the activation is the input since there is no activation function.
        preactivations.append(X)
        activations.append(X)

        # 2. Hidden layers:
        # Preactivation is defined by Z_i = W_{i-1}A_{i-1} + B_{i-1} where i is the ith layer.
        # Activation is defined by A_i = f_i(Z_i) where f_i is the activation function in the ith layer applied elementwise.
        for l in range(1, len(self.layer_sizes) - 1):
            preactivations.append(np.dot(activations[-1], self.weights[l-1]) + self.biases[l-1])
            activations.append(self.relu(preactivations[-1]))
        
        # 3. Output layer: Similar to the hidden layer, but the activation function depends on the type of problem.
        #   1. For binary classification, like make_moons, sigmoid is used.
        #   2. For multiclass, softmax is used.
        #   3. For regression, identity (f(x)=x) or relu is used.

        preactivations.append(np.dot(activations[-1], self.weights[(len(self.layer_sizes) - 1) - 1]) + self.biases[(len(self.layer_sizes) - 1)-1])
        activations.append(self.sigmoid(preactivations[-1]))
        
        assert (len(preactivations) == len(activations))

        # TODO: Implement activation layer of the output layer based on the task. Currently, the task is binary classification using sigmoid.

        return (preactivations, activations)
    

    def backward_propagation(self, preactivations, activations, X, y):
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # TODO: Backward propagation 
        

        return dW, db
        

    def calculate_updated_weights_and_biases(self, X, y):
        preactivations, activations = self.forward_propagation(X)
        return self.backward_propagation(preactivations, activations, X, y)
    
    def fit(self, X, y):
        n_input = X.shape[0]
        n_batches = max(n_input // self.batch_size, 1)
        
        # For every iteration/epoch,
        for i in range(self.max_iter):
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
                weights_update = [np.zeros_like(w) for w in self.weights]
                biases_update = [np.zeros_like(b) for b in self.biases]
                for X_inp, y_inp in zip(X_batch, y_batch):
                    current_weights_update, current_biases_update = self.calculate_updated_weights_and_biases(X_inp, y_inp)
                    weights_update = [w1 + w2 for w1, w2 in zip(weights_update, current_weights_update)]
                    biases_update = [w1 + w2 for w1, w2 in zip(biases_update, current_biases_update)]

                # Update weights and biases
                self.weights = [w1 - self.learning_rate * w2 for w1, w2 in zip(self.weights, weights_update)]
                self.biases = [w1 - self.learning_rate * w2 for w1, w2 in zip(self.biases, biases_update)]
            
    
    def predict(self, X):
        _, activations = self.forward_propagation(X)
        L = len(self.layer_sizes) - 1
        y_pred = activations[-1] >= 0.5
        return y_pred.astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Train the Neural Network
nn = NeuralNetwork(
    layer_sizes=[2, 32, 16, 1],  
    learning_rate=0.03,          
    max_iter=200,               
    batch_size=64,
    verbose=False
)
nn.fit(X_train, y_train)


# Evaluate Neural Network model
y_pred_nn = nn.predict(X_test)
accuracy_autograd = accuracy_score(y_test, y_pred_nn)
print(f"Autograd Neural Network Accuracy: {accuracy_autograd:.4f}")

# Train scikit-learn MLP
sklearn_nn = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu', 
    solver='sgd',
    learning_rate_init=0.03,
    max_iter=200,
)
sklearn_nn.fit(X_train, y_train)

# Evaluate scikit-learn model
y_pred_sklearn = sklearn_nn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"scikit-learn MLPClassifier Accuracy: {accuracy_sklearn:.4f}")

# Plot decision boundaries
def plot_decision_boundary(model, X, y, title, ax):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    if title.startswith('Neural Network'):
        Z = model.predict(np.array(np.c_[xx.ravel(), yy.ravel()]))
        Z = np.array(Z).reshape(xx.shape)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

# Print stats
print("\nConfusion Matrix - Autograd NN:")
print(confusion_matrix(y_test, y_pred_nn))
print("\nConfusion Matrix - scikit-learn MLP:")
print(confusion_matrix(y_test, y_pred_sklearn))

print("\nComparison Summary:")
print(f"Autograd NN accuracy: {accuracy_autograd:.4f}")
print(f"scikit-learn MLP accuracy: {accuracy_sklearn:.4f}")
print(f"Accuracy difference: {abs(accuracy_autograd - accuracy_sklearn):.4f}")