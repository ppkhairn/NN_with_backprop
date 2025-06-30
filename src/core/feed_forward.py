import os
import numpy as np
from src.core.layers import NeuNet
from src.utils.activations import sigmoid, relu, tanh

activation_funcs = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh
}

class FeedForward(NeuNet):

    def __init__(self):
        super().__init__()

    def forward_pass(self):
        
        for i in range(len(self.layers)-1):
            self.layers[i+1] = self.weights[i] @ self.layers[i] + self.biases[i]
            ac_func = activation_funcs[self.activations_layers[i]]
            self.layers[i+1] = ac_func(self.layers[i+1])
