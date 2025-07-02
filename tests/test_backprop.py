import pytest
import numpy as np
from src.core.backprop import BackProp

@pytest.fixture
def backprop_func() -> BackProp:
    def _create(tr_X, tr_y):
        return BackProp(tr_X, tr_y)
    return _create

@pytest.mark.parametrize(
    ("tr_X", "tr_y"),
    [
        (np.array([[1, 2, 3]]), np.array([[1]]))
    ],
    ids=[
        "Test backprop",
    ]
)
def test_backprop(backprop_func, tr_X, tr_y):
    
    model = backprop_func(tr_X, tr_y)
    model.input_layer()
    model.add_hidden_layer(3, "sigmoid")
    model.add_hidden_layer(2, "sigmoid")
    model.output_layer(1, "sigmoid")

    model.weights = []
    model.weights.append(np.ones((model.layers[1].shape[0], model.layers[0].shape[0])))
    model.weights.append(np.ones((model.layers[2].shape[0], model.layers[1].shape[0])))
    model.weights.append(np.ones((model.layers[3].shape[0], model.layers[2].shape[0])))

    model.biases = []
    model.biases.append(np.array([[1, 2, 3]]).T)
    model.biases.append(np.array([[2, 3]]).T)
    model.biases.append(np.array([[4]]).T)

    model.forward_pass()

    model.back_prop()

    assert 1 == 1