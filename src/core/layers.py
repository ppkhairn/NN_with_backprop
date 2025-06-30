# import Packages

import os
import numpy as np
import logging
from time import time
import pandas as pd
from src.utils.parameters_init import xavier_init
from typing import Literal

ActivationType = Literal["relu", "sigmoid", "tanh"]

class NeuNet():

    def __init__(self):
        self.layers = []
        self.activations_layers = []
        # self.labels = labels
    
    def input_layer(self, tr_data: np.ndarray):
        self.layers.append(np.array(tr_data))

    def add_hidden_layer(self, neurons: int, activation: ActivationType):
        self.layers.append(np.zeros(neurons))
        self.activations_layers.append(activation)
    
    def output_layer(self, neurons: int, activation: ActivationType):
        self.layers.append(np.zeros(neurons))
        self.activations_layers.append(activation)
    
    def initialize_weights(self):
        self.shape = [(self.layers[i+1].shape[0], self.layers[i].shape[0]) for i in range(len(self.layers)-1)]
        self.weights = [xavier_init(i) for i in self.shape]
        self.biases = [xavier_init((i[0],)) for i in self.shape]
        
    
