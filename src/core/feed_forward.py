import os
import numpy as np
from src.core.layers import NeuNet
from src.utils.activations import sigmoid, relu, tanh

activation_funcs = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh
}

class FeedForward():

    def __init__(self, net: NeuNet):
        self.net = net

    def forward_pass(self) -> None:
        
        for i in range(len(self.net.layers)-1):
            if self.net.layers[i].ndim != 2:
                self.net.layers[i] = self.net.layers[i].reshape(self.net.tr_data.shape[1], 1)
            self.net.layers[i+1] = self.net.weights[i] @ self.net.layers[i] + self.net.biases[i]
            ac_func = activation_funcs[self.net.activations_layers[i]]
            self.net.layers[i+1] = ac_func(self.net.layers[i+1])
            
            