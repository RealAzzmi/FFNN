from neural_network import NeuralNetwork, relu, sigmoid, binary_cross_entropy

LAYER_SIZES = [784, 128, 64, 10]
HIDDEN_LAYER_ACTIVATIONS = [relu, relu]
OUTPUT_LAYER_ACTIVATION = sigmoid
LOSS_FUNCTION = binary_cross_entropy
LEARNING_RATE = 0.001
MAX_ITER = 50
BATCH_SIZE = 256
OPTIMIZER = 'adam'
INITIALIZATION_METHOD = 'he'
VERBOSE = True