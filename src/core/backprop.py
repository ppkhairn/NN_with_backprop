import os
import numpy as np
from src.core.feed_forward import FeedForward
from src.utils.activations import sigmoid, relu, tanh
from src.utils.derivatives import diff_sigmoid, diff_binary_cross_entropy

activation_funcs = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh
}

class BackProp(FeedForward):

    def __init__(self):
        super().__init__()
        self.deltas = [np.zeros_like(s) for s in self.layers]
        del self.deltas[0]
        self.diff_losses_weights = [np.zeros_like(s) for s in self.weights]
        self.diff_losses_biases = [np.zeros_like(s) for s in self.biases]

    def back_prop(self, label):
        self.del_final = diff_sigmoid(self.layers[-1]) * diff_binary_cross_entropy(self.layers[-1], label)
        self.deltas.append(self.del_final)
        for i in range(len(self.layers)-1, 1, -1):
            self.deltas.append((self.weights[i-1] @ self.deltas[-1].T) * diff_sigmoid(self.layers[i-1]))
        self.deltas = self.deltas[::-1]


    def derivatives_(self):
        for i in range(len(self.deltas)):
            der_w = self.layers[i] @ self.del_final[i].T
            self.diff_losses_weights.append(der_w)
            der_b = self.del_final[i].T
            self.diff_losses_biases.append(der_b)

    