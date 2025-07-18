import pytest
import numpy as np
from src.nn_with_backprop_pk.core.layers import NeuNet

# @pytest.fixture
# def nn(request) -> NeuNet:
#     tr_X, tr_y = request.param
#     return NeuNet(tr_X, tr_y)


@pytest.mark.parametrize(
        ("tr_X", "tr_y"),
        [
            (np.array([[1, 2, 3]]), np.array([[1]]))
        ],
        # indirect=True, # Because we want pytest to pass the test parameters ((tr_X, tr_y)) to the fixture, not to the test function directly.
        ids=[
            "Test NN architecture"
        ]

)

def test_nn_arch(tr_X, tr_y):
    model = NeuNet(tr_X, tr_y)
    model.input_layer()
    model.add_hidden_layer(2, "sigmoid")
    model.add_hidden_layer(2, "sigmoid")
    model.output_layer(1, "sigmoid")

    assert len(model.layers) == 4
    assert (model.layers[0].shape, model.layers[1].shape, 
            model.layers[2].shape, model.layers[3].shape) == ((3,1), (2,1), (2,1), (1,1))
    

@pytest.mark.parametrize(
        ("tr_X", "tr_y"),
        [
            (np.array([[1, 2, 3]]), np.array([[1]]))
        ],
        # indirect=True, # Because we want pytest to pass the test parameters ((tr_X, tr_y)) to the fixture, not to the test function directly.
        ids=[
            "Test NN Parameters"
        ]

)
    
def test_parameters(tr_X, tr_y):
    model = NeuNet(tr_X, tr_y)
    model.input_layer()
    model.add_hidden_layer(2, "sigmoid")
    model.add_hidden_layer(2, "sigmoid")
    model.output_layer(1, "sigmoid")

    model.initialize_weights()

    assert (model.weights[0].shape,
            model.weights[1].shape,
            model.weights[2].shape) == ((2, 3), (2, 2), (1, 2))
    assert (model.biases[0].shape,
            model.biases[1].shape,
            model.biases[2].shape) == ((2,1), (2,1), (1,1))
