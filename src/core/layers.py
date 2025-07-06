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

    def __init__(self, tr_X: np.ndarray, tr_y:np.ndarray):
        self.layers = []
        self.activations_layers = []
        self.tr_data = tr_X
        self.label = tr_y
    
    def input_layer(self) -> None:
        self.layers.append(self.tr_data[0].reshape(self.tr_data.shape[1],1))

    def add_hidden_layer(self, neurons: int, activation: ActivationType) -> None:
        self.layers.append(np.zeros((neurons,1), dtype=np.float64))
        self.activations_layers.append(activation)
    
    def output_layer(self, neurons: int, activation: ActivationType) -> None:
        self.layers.append(np.zeros((neurons,1), dtype=np.float64))
        self.activations_layers.append(activation)
    
    def initialize_weights(self) -> None:
        self.shape = [(self.layers[i+1].shape[0], self.layers[i].shape[0]) for i in range(len(self.layers)-1)]
        self.weights = [xavier_init(i) for i in self.shape]
        self.biases = [xavier_init((i[0],1)) for i in self.shape]
        
    
