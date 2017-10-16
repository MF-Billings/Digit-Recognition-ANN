import math
import numpy as np

# NumPy is required to run this program


def sigmoid(n, use_derivative=False):
    # use numpy version if it's imported; just for fun
    if 'np' in globals():
        if use_derivative:
            # assume sigmoid(x) has already been evaluated and is supplied for n ie. n = sigmoid(x)
            return sigmoid(n)*(1-sigmoid(n))
        else:
            return 1.0 / (1.0 + np.exp(-n))
    else:
        print "NumPy is required to run this program"
        # return 1 / (1 + math.e ** n)


def tanh(n, use_derivative=False):
    # use numpy version if it's imported; just for fun
    if 'np' not in globals():
        print "You should really use NumPy"
    if use_derivative:
        return 1 - math.tanh(n) ** 2
    else:
        return math.tanh(n)
