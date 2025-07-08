import pytest
import numpy as np
from src.core.feed_forward import FeedForward#, NeuNet
from src.core.layers import NeuNet

@pytest.mark.parametrize(
        ("tr_X", "tr_y"),
        [
            (np.array([[1, 2, 3]]), np.array([[1]]))
        ],
        ids=[
            "Test feed forward loop"
        ]

)
def test_ff(tr_X, tr_y):
    net = NeuNet(tr_X, tr_y)
    net.input_layer()
    net.add_hidden_layer(3, "sigmoid")
    net.add_hidden_layer(2, "sigmoid")
    net.output_layer(1, "sigmoid")

    net.weights = []
    net.weights.append(np.ones((net.layers[1].shape[0], net.layers[0].shape[0])))
    net.weights.append(np.ones((net.layers[2].shape[0], net.layers[1].shape[0])))
    net.weights.append(np.ones((net.layers[3].shape[0], net.layers[2].shape[0])))

    net.biases = []
    net.biases.append(np.array([[1, 2, 3]]).T)
    net.biases.append(np.array([[2, 3]]).T)
    net.biases.append(np.array([[4]]).T)

    model = FeedForward(net)
    model.forward_pass(tr_X[0])

    expected = np.array([0.99750464]) #0.99750464
    assert np.allclose(net.layers[-1], expected, rtol=1e-5)
