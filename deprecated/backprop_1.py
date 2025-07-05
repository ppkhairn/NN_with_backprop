import os
import numpy as np
from core.feed_forward import FeedForward
from src.utils.activations import sigmoid, relu, tanh
from src.utils.derivatives import diff_sigmoid, diff_binary_cross_entropy

activation_funcs = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh
}

class BackProp():

    def __init__(self, forward_pass: FeedForward, learning_rate: float):
        self.ff = forward_pass #not forward_pass() since "()" creates an instance, but we just need to pass an instance while using these classes
        self.learning_rate = learning_rate

    def back_prop(self) -> None:
        self.deltas_ = [np.zeros_like(s) for s in self.ff.net.layers]
        del self.deltas_[0]
        self.deltas = []
        self.diff_losses_weights_ = [np.zeros_like(s) for s in self.ff.net.weights]
        self.diff_losses_biases_ = [np.zeros_like(s) for s in self.ff.net.biases]
        self.diff_losses_weights = []
        self.diff_losses_biases = []

        self.del_final = diff_sigmoid(self.ff.net.layers[-1]) * diff_binary_cross_entropy(self.ff.net.layers[-1], self.ff.net.label.reshape(self.ff.net.label.shape[1],1))
        self.deltas.append(self.del_final)
        for i in range(len(self.ff.net.layers)-1, 1, -1):
            self.deltas.append((self.ff.net.weights[i-1].T @ self.deltas[-1]) * diff_sigmoid(self.ff.net.layers[i-1]))
        self.deltas = self.deltas[::-1]
        for i in range(len(self.deltas)):
            # der_w = self.layers[i] @ self.deltas[i].T
            der_w = self.deltas[i] @ self.ff.net.layers[i].T
            self.diff_losses_weights.append(der_w)
            der_b = self.deltas[i]
            self.diff_losses_biases.append(der_b)
        
        return self.diff_losses_weights, self.diff_losses_biases
    
    def update_parameters(self):
        for i in range(len(self.ff.net.weights)):
            self.ff.net.weights[i] -= self.learning_rate * self.diff_losses_weights[i]
            self.ff.net.biases[i] -= self.learning_rate * self.diff_losses_biases[i]
        return self.ff.net.weights, self.ff.net.biases

    

    