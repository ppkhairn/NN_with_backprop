import pytest
import numpy as np
from src.core.backprop import BackProp
from src.core.layers import NeuNet
from src.core.feed_forward import FeedForward
from src.utils.loss_functions import binary_cross_entropy, mean_squared_error

@pytest.mark.parametrize(
    ("tr_X", "tr_y", "learning_rate", "loss_function"),
    [
        (np.array([[1, 2, 3]]), np.array([[1]]), 0.01, "binary_cross_entropy"), 
        (np.array([[1, 2, 3]]), np.array([[1]]), 0.01, "mean_squared_error")
    ],
    ids=[
        "Test backprop", "with mse"
    ]
)
def test_backprop(tr_X, tr_y, learning_rate, loss_function):
    
    net = NeuNet(tr_X, tr_y)
    net.input_layer()
    net.add_hidden_layer(3, "sigmoid")
    net.add_hidden_layer(2, "sigmoid")
    net.output_layer(1, "sigmoid")

    net.weights = []
    net.weights.append(np.ones((net.layers[1].shape[0], net.layers[0].shape[0]), dtype=np.float64))
    net.weights.append(np.ones((net.layers[2].shape[0], net.layers[1].shape[0]), dtype=np.float64))
    net.weights.append(np.ones((net.layers[3].shape[0], net.layers[2].shape[0]), dtype=np.float64))

    net.biases = []
    net.biases.append(np.array([[1, 2, 3]], dtype=np.float64).T)
    net.biases.append(np.array([[2, 3]], dtype=np.float64).T) #, dtype=np.float64
    net.biases.append(np.array([[4]], dtype=np.float64).T)

    ff = FeedForward(net)
    ff.forward_pass()

    bp = BackProp(ff, learning_rate, loss_function)
    # bp.back_prop()
    # weights, biases = bp.update_parameters()

    if loss_function == "binary_cross_entropy":
        bp.back_prop()
        weights, biases = bp.update_parameters()
        assert np.allclose(weights[-1], np.array([[1.00002479, 1.00002489]]))
        assert np.allclose(biases[-1], np.array([[4.00002495]]))
    
    elif loss_function == "mean_squared_error":
        # Expect NotImplementedError for this loss
        with pytest.raises(NotImplementedError):
            bp.back_prop()