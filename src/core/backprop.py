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
        self.deltas = []

    def back_prop(self, label):
        self.del_final = diff_sigmoid(self.layers[-1]) * diff_binary_cross_entropy(self.layers[-1], label)
        for i in range(len(self.layers)-2-1):
            d_h = (self.weights[] @ self.del_final) * diff_sigmoid(self.layers[])


    def derivatives_(self):
        der_w = self.layers[-2] @ self.del_final

    