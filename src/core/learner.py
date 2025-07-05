import os
import numpy as np
from src.core.backprop import BackProp
from src.utils.activations import sigmoid, relu, tanh
from src.utils.derivatives import diff_sigmoid, diff_binary_cross_entropy
from src.utils.loss_functions import binary_cross_entropy

class Learner(BackProp):

    def __init__(self, backprop: BackProp, epoch: int):
        
        self.bp = backprop
        self.epoch = epoch

    def calculate_avg_loss(self) -> float:
        list_loss = []
        for i in range(len(self.bp.ff.net.tr_data)):
            _loss = binary_cross_entropy(self.bp.ff.net.tr_data[i], self.bp.ff.net.label[i])
            list_loss.append(_loss)
        avg_loss = sum(list_loss) / len(list_loss)

        return avg_loss
    
    def train(self):
        loss_epoch_avg = []
        for i in range(self.epoch):
            loss_epoch = []
            for j in range(len(self.bp.ff.net.tr_data)):

                self.bp.ff.forward_pass()
                self.bp.back_prop()
                self.bp.update_parameters()
                loss_epoch.append(self.calculate_avg_loss())
            loss_epoch_avg.append(sum(loss_epoch) / len(loss_epoch))
        
        return loss_epoch_avg
    
    