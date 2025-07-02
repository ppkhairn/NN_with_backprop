import pytest
import numpy as np
from src.core.backprop import BackProp

@pytest.fixture
def backprop_func() -> BackProp:
    def _create(tr_X, tr_y):
        return BackProp(tr_X, tr_y)
    return _create

@pytest.mark.parametrize(
    ("tr_X", "tr_y", "learning_rate"),
    [
        (np.array([[1, 2, 3]]), np.array([[1]]), 0.01)
    ],
    ids=[
        "Test backprop",
    ]
)
def test_backprop(backprop_func, tr_X, tr_y, learning_rate):
    
    model = backprop_func(tr_X, tr_y)
    model.input_layer()
    model.add_hidden_layer(3, "sigmoid")
    model.add_hidden_layer(2, "sigmoid")
    model.output_layer(1, "sigmoid")

    model.weights = []
    model.weights.append(np.ones((model.layers[1].shape[0], model.layers[0].shape[0]), dtype=np.float64))
    model.weights.append(np.ones((model.layers[2].shape[0], model.layers[1].shape[0]), dtype=np.float64))
    model.weights.append(np.ones((model.layers[3].shape[0], model.layers[2].shape[0]), dtype=np.float64))

    model.biases = []
    model.biases.append(np.array([[1, 2, 3]], dtype=np.float64).T)
    model.biases.append(np.array([[2, 3]], dtype=np.float64).T) #, dtype=np.float64
    model.biases.append(np.array([[4]], dtype=np.float64).T)

    model.forward_pass()

    model.back_prop()

    weights, biases = model.update_parameters(learning_rate)

    assert np.allclose(weights[-1], np.array([[1.00002479, 1.00002489]]))
    assert np.allclose(biases[-1], np.array([[4.00002495]]))

