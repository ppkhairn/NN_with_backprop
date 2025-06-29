# import Packages

import os
import numpy as np
import logging
from time import time
import pandas as pd
from src.utils.parameters_init import xavier_init

class nn():

    def __init__(self):
        self.layers = []
        self.actications_layers = []
    
    def input_layer(self, tr_data):
        self.layers.append(np.zeros(tr_data.shape(0)))

    def add_hidden_layer(self, neurons, activation):
        self.layers.append(np.zeros(neurons))
        self.actications_layers.append(activation)
    
    def output_layer(self):
        self.layers.append(np.zeros(1))
    
    def initialize_weights(self):
        self.shape = [(self.layers[i].shape[0], self.layers[i+1].shape[0]) for i in range(len(self.layers)-1)]
        self.weights = [xavier_init(i) for i in self.shape]
        self.biases = [xavier_init((i[1],)) for i in self.shape]
    
    # def feed_forward(self):
