import pytest
import numpy as np
from src.core.feed_forward import FeedForward, NeuNet

# @pytest.fixture
# def nn() -> NeuNet:
#     return NeuNet()

@pytest.fixture
def feed_forward() -> FeedForward:
    return FeedForward()

def test_ff(feed_forward):
    tr_Data = np.array([1, 2, 3])
    feed_forward.input_layer(tr_Data)
    feed_forward.add_hidden_layer(2, "sigmoid")
    feed_forward.add_hidden_layer(2, "sigmoid")
    feed_forward.output_layer(1, "sigmoid")

    feed_forward.weights = []
    feed_forward.weights.append(np.ones((2, 3)))
    feed_forward.weights.append(np.ones((2, 2)))
    feed_forward.weights.append(np.ones((1, 2)))

    feed_forward.biases = []
    feed_forward.biases.append(np.array([[1, 2]]).T)
    feed_forward.biases.append(np.array([[2, 3]]).T)
    feed_forward.biases.append(np.array([[4]]).T)

    feed_forward.forward_pass()

    expected = np.array([0.997465675536554])
    assert np.allclose(feed_forward.layers[-1], expected, rtol=1e-5)
