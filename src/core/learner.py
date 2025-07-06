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

    def calculate_loss(self, predicted_sample, actual_sample) -> float:
        list_loss = []
        # for i in range(len(self.bp.ff.net.layers[-1])):
        # _loss = binary_cross_entropy(self.bp.ff.net.layers[-1], self.bp.ff.net.label[0])
        _loss = binary_cross_entropy(predicted_sample, actual_sample)
            # list_loss.append(_loss)
        # avg_loss = sum(list_loss) / len(list_loss)

        return _loss
    
    def train(self):
        loss_epoch_avg = []
        for i in range(self.epoch):
            loss_epoch = []
            for j in range(len(self.bp.ff.net.tr_data)):

                self.bp.ff.forward_pass()
                self.bp.back_prop()
                self.bp.update_parameters()
                reshaped_label = self.bp.ff.net.label[j].reshape(self.bp.ff.net.label.shape[1], 1)
                loss_epoch.append(self.calculate_loss(self.bp.ff.net.layers[-1], reshaped_label))

                # Update input layer
                self.bp.ff.net.layers[0] = self.bp.ff.net.tr_data[j+1]
            loss_epoch_avg.append(sum(loss_epoch) / len(loss_epoch))
        
        return loss_epoch_avg
    
    