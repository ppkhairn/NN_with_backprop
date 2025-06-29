import pytest
import numpy as np
from src.layers import NeuNet

@pytest.fixture
def nn() -> NeuNet:
    return NeuNet()

def test_nn_arch(nn):
    tr_Data = np.array([1, 2, 3])
    nn.input_layer(tr_Data)
    nn.add_hidden_layer(2, "sigmoid")
    nn.add_hidden_layer(2, "sigmoid")
    nn.output_layer(1, "sigmoid")

    assert len(nn.layers) == 4
    assert (nn.layers[0].shape, nn.layers[1].shape, 
            nn.layers[2].shape, nn.layers[3].shape) == ((3,), (2,), (2,), (1,))
    
def test_parameters(nn):
    tr_Data = np.array([1, 2, 3])
    nn.input_layer(tr_Data)
    nn.add_hidden_layer(2, "sigmoid")
    nn.add_hidden_layer(2, "sigmoid")
    nn.output_layer(1, "sigmoid")

    nn.initialize_weights()

    assert (nn.weights[0].shape,
            nn.weights[1].shape,
            nn.weights[2].shape) == ((2, 3), (2, 2), (1, 2))
    assert (nn.biases[0].shape,
            nn.biases[1].shape,
            nn.biases[2].shape) == ((2,), (2,), (1,))
