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

    def __init__(self, tr_X, tr_y):
        super().__init__(tr_X, tr_y)
        # self.deltas = [np.zeros_like(s) for s in self.layers]
        # del self.deltas[0]
        # self.diff_losses_weights = [np.zeros_like(s) for s in self.weights]
        # self.diff_losses_biases = [np.zeros_like(s) for s in self.biases]

    def back_prop(self) -> None:
        self.deltas_ = [np.zeros_like(s) for s in self.layers]
        del self.deltas_[0]
        self.deltas = []
        self.diff_losses_weights_ = [np.zeros_like(s) for s in self.weights]
        self.diff_losses_biases_ = [np.zeros_like(s) for s in self.biases]
        self.diff_losses_weights = []
        self.diff_losses_biases = []

        self.del_final = diff_sigmoid(self.layers[-1]) * diff_binary_cross_entropy(self.layers[-1], self.label.reshape(self.label.shape[1],1))
        self.deltas.append(self.del_final)
        for i in range(len(self.layers)-1, 1, -1):
            self.deltas.append((self.weights[i-1].T @ self.deltas[-1]) * diff_sigmoid(self.layers[i-1]))
        self.deltas = self.deltas[::-1]
        for i in range(len(self.deltas)):
            # der_w = self.layers[i] @ self.deltas[i].T
            der_w = self.deltas[i] @ self.layers[i].T
            self.diff_losses_weights.append(der_w)
            der_b = self.deltas[i]
            self.diff_losses_biases.append(der_b)
        
        return self.diff_losses_weights, self.diff_losses_biases

    # def derivatives_(self):
    #     for i in range(len(self.deltas)):
    #         der_w = self.layers[i] @ self.del_final[i].T
    #         self.diff_losses_weights.append(der_w)
    #         der_b = self.del_final[i].T
    #         self.diff_losses_biases.append(der_b)

    