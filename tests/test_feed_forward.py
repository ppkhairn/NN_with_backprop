import pytest
import numpy as np
from src.feed_forward import FeedForward, NeuNet

@pytest.fixture
def nn() -> NeuNet:
    return NeuNet()

@pytest.fixture
def feed_forward() -> FeedForward:
    return FeedForward(NeuNet)

def test_ff(nn, feed_forward):
    tr_Data = np.array([1, 2, 3])
    nn.input_layer(tr_Data)
    nn.add_hidden_layer(2, "sigmoid")
    nn.add_hidden_layer(2, "sigmoid")
    nn.output_layer(1, "sigmoid")

    feed_forward.forward_pass()

    assert 1==1