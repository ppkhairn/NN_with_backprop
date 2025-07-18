import numpy as np
from src.nn_with_backprop_pk.core.backprop import BackProp
from src.nn_with_backprop_pk.utils.activations import sigmoid, relu, tanh
from src.nn_with_backprop_pk.utils.derivatives import diff_sigmoid, diff_binary_cross_entropy
from src.nn_with_backprop_pk.utils.loss_functions import binary_cross_entropy
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
# Set up logging once in your main script or module
logging.basicConfig(level=logging.INFO)

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
        self.loss_epoch_avg = []
        for i in range(self.epoch):
            loss_epoch = []
            for j in range(len(self.bp.ff.net.tr_data)):

                self.bp.ff.forward_pass(self.bp.ff.net.tr_data[j])
                d_w, d_b = self.bp.back_prop()
                # logging.info(np.linalg.norm(d_w[-1]))
                self.bp.update_parameters()
                reshaped_label = self.bp.ff.net.label[j].reshape(self.bp.ff.net.label.shape[1], 1)
                loss_epoch.append(self.calculate_loss(self.bp.ff.net.layers[-1], reshaped_label))

                # # Update input layer
                # self.bp.ff.net.layers[0] = self.bp.ff.net.tr_data[j+1]
            self.loss_epoch_avg.append(sum(loss_epoch) / len(loss_epoch))
            logging.info(f"Loss - {sum(loss_epoch) / len(loss_epoch)}")
            logging.info(f"Epoch: {str(i)} - Average Loss: {self.loss_epoch_avg[-1]}")
        
        return self.loss_epoch_avg
    
    def plot_loss(self)-> None:
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(list(range(self.epoch)), self.loss_epoch_avg)
        ax.set_title("Loss vs epoch")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Average Loss")
        plt.tight_layout()
        plt.show()

        return None
    
    def binary_classification_accuracy(self, test_data: np.ndarray, test_label: np.ndarray) -> np.float64:
        preds = []
        for i in range(test_data.shape[0]):
            # self.bp.ff.net = self.bp.ff.net(test_X, test_y)
            self.bp.ff.forward_pass(test_data[i].reshape(test_data.shape[1], 1))
            if self.bp.ff.net.layers[-1] < 0.5:
                preds.append(0)
            else: 
                preds.append(1)
        preds_arr = np.array(preds)
        if test_label.ndim > 1:
            test_label = test_label.reshape(test_label.shape[0],)
        
        return sum(test_label == preds_arr) / len(test_label)


    