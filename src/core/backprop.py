import os
import numpy as np
from typing import List, Tuple, Literal
from src.core.feed_forward import FeedForward
from src.utils.activations import sigmoid, relu, tanh
from src.utils.derivatives import diff_sigmoid, diff_binary_cross_entropy
from src.utils.loss_functions import mean_squared_error, binary_cross_entropy

activation_funcs = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh
}

loss_funcs = {
    "mean_squared_error": mean_squared_error,
    "binary_cross_entropy": binary_cross_entropy
}

LossType = Literal["mean_squared_error", "binary_cross_entropy"]

class BackProp():

    def __init__(self, forward_pass: FeedForward, learning_rate: float, loss_function: LossType):
        self.ff = forward_pass #not forward_pass() since "()" creates an instance, but we just need to pass an instance while using these classes
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def back_prop(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # self.deltas_ = [np.zeros_like(s) for s in self.ff.net.layers]
        # del self.deltas_[0]
        self.deltas = []
        # self.diff_losses_weights_ = [np.zeros_like(s) for s in self.ff.net.weights]
        # self.diff_losses_biases_ = [np.zeros_like(s) for s in self.ff.net.biases]
        self.diff_losses_weights = []
        self.diff_losses_biases = []

        if self.ff.net.activations_layers[-1] == "sigmoid" and self.loss_function == "binary_cross_entropy":
            # Special case: when using sigmoid activation + binary cross-entropy loss,
            # the gradient simplifies to (y_pred - y_true)
            # Look at the docs for this special case
            self.del_final = self.ff.net.layers[-1] - self.ff.net.label[0].reshape(self.ff.net.label.shape[1],1) 
        else: 
            #Example - self.del_final = diff_sigmoid(self.ff.net.layers[-1]) * diff_binary_cross_entropy(self.ff.net.layers[-1], self.ff.net.label.reshape(self.ff.net.label.shape[1],1))

            raise NotImplementedError(
                f"Delta not implemented for combination:\n"
                f"  Loss function     : {self.loss_function}\n"
                f"  Output activation : {self.ff.net.activations_layers[-1]}"
            ) 
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
    
    def update_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        for i in range(len(self.ff.net.weights)):
            self.ff.net.weights[i] -= self.learning_rate * self.diff_losses_weights[i]
            self.ff.net.biases[i] -= self.learning_rate * self.diff_losses_biases[i]
        return self.ff.net.weights, self.ff.net.biases

    

    