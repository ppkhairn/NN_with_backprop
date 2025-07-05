import os
import numpy as np
from core.backprop import BackProp
from src.utils.activations import sigmoid, relu, tanh
from src.utils.derivatives import diff_sigmoid, diff_binary_cross_entropy
from src.utils.loss_functions import binary_cross_entropy

class Learner(BackProp):

    def __init__(self, tr_X, tr_y, learning_rate):
        super().__init__(tr_X, tr_y, learning_rate=learning_rate)

    def calculate_avg_loss(self, tr_X, tr_y) -> float:
        list_loss = []
        for i in range(len(tr_X)):
            _loss = binary_cross_entropy(tr_X[i], tr_y[i])
            list_loss.append(_loss)
        avg_loss = sum(list_loss) / len(list_loss)

        return avg_loss
    
    def train(self, tr_X, tr_y, epoch):
        
        for i in range(epoch):
            loss_epoch = []
            for j in range(len(tr_X)):

                self.forward_pass()
                self.back_prop()
                self.update_parameters()
                loss_epoch.append(self.calculate_avg_loss())
        
        return sum(loss_epoch) / len(loss_epoch)
    
    