import pytest
import numpy as np
from core.feed_forward import FeedForward#, NeuNet
from core.layers import NeuNet

# @pytest.fixture
# def nn(request) -> NeuNet:
#     tr_X, tr_y = request.param
#     return NeuNet(tr_X, tr_y)

@pytest.fixture
def feed_forward_func(request) -> FeedForward:
    tr_X, tr_y = request.param
    net = NeuNet(tr_X, tr_y)
    return FeedForward(net)

@pytest.mark.parametrize(
        "feed_forward_func",
        [
            (np.array([[1, 2, 3]]), np.array([[1]]))
        ],
        indirect=True,
        ids=[
            "Test feed forward loop"
        ]

)
def test_ff(feed_forward_func):
    model = feed_forward_func
    model.net.input_layer()
    model.net.add_hidden_layer(3, "sigmoid")
    model.net.add_hidden_layer(2, "sigmoid")
    model.net.output_layer(1, "sigmoid")

    model.net.weights = []
    model.net.weights.append(np.ones((model.net.layers[1].shape[0], model.net.layers[0].shape[0])))
    model.net.weights.append(np.ones((model.net.layers[2].shape[0], model.net.layers[1].shape[0])))
    model.net.weights.append(np.ones((model.net.layers[3].shape[0], model.net.layers[2].shape[0])))

    model.net.biases = []
    model.net.biases.append(np.array([[1, 2, 3]]).T)
    model.net.biases.append(np.array([[2, 3]]).T)
    model.net.biases.append(np.array([[4]]).T)

    model.forward_pass()

    expected = np.array([0.99750464]) #0.99750464
    assert np.allclose(model.net.layers[-1], expected, rtol=1e-5)
