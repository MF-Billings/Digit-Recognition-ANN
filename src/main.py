"""
Matt Billings
Feb 2017
NOTE: NumPy is required to run this program
"""
import logging
import traceback

import numpy as np

import ann_io
import training
import my_math
import config


# REMOVE
# set up logging functionality
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# layer the specific layer
# derivative_supplied the neurons in layer already contain the derivative for each neuron. Since the derivative is
# already calculated and stored for error propagation it saves time to just plug that value in place of activation(x)
def activation_function(layer, function=my_math.sigmoid, use_derivative=False, derivative_supplied=True):
    clone = np.zeros(len(layer))
    if use_derivative and derivative_supplied:
        if function == my_math.sigmoid:
            clone = np.array([i * (1-i) for i in layer])
        elif function == my_math.tanh:
            clone = np.array([1 - i ** 2 for i in layer])
        else:
            logger.debug("Unrecognized activation function provided: %s", function)
            traceback.print_stack()
    else:
        for i in xrange(len(layer)):
            clone[i] = function(layer[i], use_derivative)

    return clone


# main entry point for the program
def main():
    # full_input = ann_io.read_digit_file("../a1digits/digit_train_0.txt")
    training_file_paths = [
        "../../a1digits/digit_train_0.txt", "../../a1digits/digit_train_1.txt",
        "../../a1digits/digit_train_2.txt", "../../a1digits/digit_train_3.txt",
        "../../a1digits/digit_train_4.txt", "../../a1digits/digit_train_5.txt",
        "../../a1digits/digit_train_6.txt", "../../a1digits/digit_train_7.txt",
        "../../a1digits/digit_train_8.txt", "../../a1digits/digit_train_9.txt"
    ]

    testing_file_paths = [
        "../../a1digits/digit_test_0.txt", "../../a1digits/digit_test_1.txt", "../../a1digits/digit_test_2.txt",
        "../../a1digits/digit_test_3.txt", "../../a1digits/digit_test_4.txt", "../../a1digits/digit_test_5.txt",
        "../../a1digits/digit_test_6.txt", "../../a1digits/digit_test_7.txt", "../../a1digits/digit_test_8.txt",
        "../../a1digits/digit_test_9.txt"
    ]

    training_set = ann_io.format_data(training_file_paths, config.random)
    testing_set = ann_io.format_data(testing_file_paths)

    # ANN parameters
    activator = my_math.sigmoid
    epochs = 32
    epsilon = 1e-6  # determines when the algorithm should stop early
    k = 10

    hidden_layer_args = [57]
    momentum = 0.9
    learning_rate = 0.5

    ann = FeedForwardNetwork(64, hidden_layer_args, k, momentum, learning_rate)
    trainer = training.Trainer()
    # using holdout
    trainer.run(ann, training_set, epochs, epsilon, activator)
    trainer.test(ann, testing_set)

    # name files based on parameters
    common_str = "_%de_%dn_%.2fm_%.2flr_%s.csv" % (epochs, hidden_layer_args[0], momentum, learning_rate, activator.__name__)
    training_filename = "trn" + common_str
    validation_filename = "vld" + common_str
    testing_filename = "tes" + common_str
    # write output to files
    ann_io.write_to_file(training_filename, trainer.get_results("training"))
    ann_io.write_to_file(validation_filename, trainer.get_results("validation"))
    ann_io.write_to_file(testing_filename, trainer.get_results("testing"))
    pass


# nodes_in_layer a list representing the number of hidden nodes in each layer
class FeedForwardNetwork:
    def __init__(self, n_inputs, nodes_in_layer, n_outputs, momentum=0.0, learning_rate=0.5,
                 bias_shared_in_layer=True):
        self.momentum = momentum
        self.learning_rate = learning_rate

        # neuron/node layers
        self.inputs = np.zeros([n_inputs])    # container for the current training pattern. Capture size of an input pattern for weight initialization

        self.hidden_layers = {}
        self.outputs = np.zeros(n_outputs)
        self.expected_outputs = None   # np.array([1 if inputs[0].expected_value == i else 0 for i in xrange(10)])

        # weight matrices
        self.weights = {}                       # if there are n layers total the size of the matrix is n-1
        self.previous_delta_weights = {}        # for momentum

        # bias matrices
        self.biases = {}

        self.e_hidden = {}
        self.e_sum_hidden = None
        self.e_outputs = np.zeros(n_outputs)    #np.zeros(n_outputs)   # np.zeros((n_outputs, 1))
        self.e_sum_output = 0    # for layer-wide error sum; not sure if I need this or individual errors
        self.deltas = {}

        # inclusive on both ends; might need to tweak scaling arithmetic to include upper bound
        self.weight_range = {
            "max": 0.5,
            "min": -0.5
        }

        # initialize layers

        # create dictionary in case of multiple layers
        self.x_hidden_layers = {}    # raw inputs before transfer function is applied
        for layer in xrange(len(nodes_in_layer)):
            self.x_hidden_layers[layer] = np.zeros(nodes_in_layer[layer]) # n_hiddens

        self.hidden_layers = {}
        for layer in xrange(len(nodes_in_layer)):
            self.hidden_layers[layer] = np.zeros(nodes_in_layer[layer])     # n_hiddens
            self.e_hidden[layer] = np.zeros(nodes_in_layer[layer])     # np.zeros((nodes_in_layer[layer], 1))

        self.x_outputs = np.zeros(n_outputs)    # inputs to the output neurons, represented by x

        self.initialize_weights(bias_shared_in_layer)

        # copy dimensions of weight matrix (I think I need the same dimensions)
        for key in self.weights:
            self.previous_derivatives[key] = np.zeros([self.weights[key].shape[0], self.weights[key].shape[1]])

        # set dimensions for deltas.  The shape and number of the delta matrices will be the same as the weight matrices
        for layer in xrange(len(self.weights)):
            self.deltas[layer] = np.zeros(self.weights[layer].shape[1])  # np.zeros((nodes_in_layer[layer], 1))

    def forward(self, activation_fn):
        # pass input through hidden layer(s)
        for layer in xrange(len(self.hidden_layers)):
            # input layer to hidden layer
            if layer == 0:
                self.x_hidden_layers[layer] = np.dot(self.inputs, self.weights[layer])
            # hidden layer to hidden layer
            else:
                self.x_hidden_layers[layer] = np.dot(self.hidden_layers[layer-1], self.weights[layer])
            self.x_hidden_layers[layer] += self.biases[layer]
            self.hidden_layers[layer] = activation_function(self.x_hidden_layers[layer], activation_fn)

        # pass from last hidden layer to output
        self.x_outputs = np.dot(self.hidden_layers[len(self.hidden_layers)-1], self.weights[len(self.weights)-1])
        self.x_outputs += self.biases[len(self.biases)-1]
        # note that the sigmoid function will always be used on the output neurons to squash the output between 0 and 1
        self.outputs = activation_function(self.x_outputs)

        # calculate error
        self.e_sum_output = 0  # the total error for the layer
        for n in xrange(len(self.outputs)):
            self.e_outputs[n] = 0.5 * np.power(self.expected_outputs[n] - self.outputs[n], 2)
            self.e_sum_output += self.e_outputs[n]

    # distribute error back through the network and update the weights
    def backward(self, activation_fn, variant=None):

        # calculate deltas (derivatives plus a bit extra) between hidden layer and output
        # not updating the weights yet, just storing the delta
        self.deltas[len(self.deltas) - 1] = activation_function(self.outputs, my_math.sigmoid, True) * (self.outputs - self.expected_outputs)
        logging.debug("%s" % self.deltas[len(self.deltas) - 1])

        # calculate other deltas separately for clarity's sake
        logging.debug("Hidden deltas")

        for layer in reversed(xrange(len(self.deltas)-1)):
            # calculated weighted sum for error
            for i in xrange(len(self.deltas[layer])):
                self.deltas[layer][i] = np.sum(self.deltas[layer+1] * self.weights[1][i])
            self.deltas[layer] *= activation_function(self.hidden_layers[layer], activation_fn, True)
            logger.debug("%s", self.deltas[layer])

        self.update_weights(variant)

    def scale_weights(self, weight_matrix, n_min, n_max):
        for row in xrange(len(weight_matrix)):
            for col in xrange(len(weight_matrix[row])):
                weight_matrix[row][col] = n_max + weight_matrix[row][col] * (n_min - n_max)

    def update_weights(self, variant=None):
        logging.debug("UPDATING WEIGHTS")

        for layer in reversed(xrange(len(self.weights))):
            if layer == 0:
                preceding_layer = self.inputs   # preceding the current layer in the backward pass (ie. to the right)
            else:
                preceding_layer = self.hidden_layers[layer - 1]

            # update weights
            for i in xrange(len(self.weights[layer])):
                for j in xrange(len(self.weights[layer][i])):
                    if variant == config.quickprop:
                        self.previous_derivatives[layer][i][j] = self.deltas[layer][j] * preceding_layer[i]           # self.deltas[len(self.deltas) - 1]
                    change_in_weight = self.learning_rate * self.deltas[layer][j] * preceding_layer[i]      # keep record of old weights for momentum

                    self.weights[layer][i][j] -= change_in_weight + self.momentum * self.previous_delta_weights[layer][i][j]
                    self.previous_delta_weights[layer][i][j] = change_in_weight                             # keep record of old weights for momentum

                    logging.debug(self.weights[layer][i][j])

    """
    create weight matrix and initialize weight matrix
    """
    def initialize_weights(self, bias_shared_in_layer):
        # real code
        np.random.seed(config.random_seed)

        last_hidden_layer = len(self.hidden_layers) - 1
        number_of_inputs = np.size(self.inputs)
        number_of_hidden_nodes = len(self.hidden_layers[0])

        # weights b/t input and 1st hidden layer
        self.weights[0] = np.random.rand(number_of_inputs, number_of_hidden_nodes)
        self.previous_delta_weights[0] = np.zeros((number_of_inputs, number_of_hidden_nodes))

        # create weight matrix for edges between hidden layers
        # number of weight matrices between hidden layers is len(hidden layers) - 1
        for layer in xrange(len(self.hidden_layers) - 1):
            # dim = nodes in layer, nodes in next layer
            self.weights[layer+1] = np.random.rand(len(self.hidden_layers[layer]), len(self.hidden_layers[layer+1]))
            self.previous_delta_weights[layer+1] = np.zeros((len(self.hidden_layers[layer]), len(self.hidden_layers[layer+1])))

        # number_of_outputs = np.size(self.outputs)
        self.weights[len(self.weights)] = np.random.rand(len(self.hidden_layers[last_hidden_layer]), np.size(self.outputs))
        self.previous_delta_weights[len(self.previous_delta_weights)] = np.zeros((len(self.hidden_layers[last_hidden_layer]), np.size(self.outputs)))

        for i in xrange(len(self.hidden_layers)):
            if bias_shared_in_layer:
                rand = scale_to_range(np.random.rand(), self.weight_range["min"], self.weight_range["max"])
                self.biases[i] = np.array([rand])
            else:
                self.biases[i] = np.random.rand(len(self.hidden_layers[i]))
        if bias_shared_in_layer:
            self.biases[len(self.biases)] = np.array([scale_to_range(np.random.rand(), self.weight_range["min"], self.weight_range["max"])])
        else:
            self.biases[len(self.biases)] = np.random.rand(len(self.outputs))

        self.scale_weights(self.weights, self.weight_range["min"], self.weight_range["max"])


def scale_to_range(n, low, high):
    return high + n * (low - high)

# program is executed by itself ie. not imported by another program
if __name__ == '__main__':
    main()
