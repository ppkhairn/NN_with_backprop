import pytest
import numpy as np
from src.core.learner import Learner
from src.core.layers import NeuNet
from src.core.feed_forward import FeedForward
from src.core.backprop import BackProp

@pytest.mark.parametrize(
        ("tr_X", "tr_y", "test_X", "test_y"),
        [
            (np.array([[1, 2, 3, 4]]), 
             np.array([[1]]), 
             np.array([[ 2.41079286,  1.86917728,  1.54099762,  2.28200427],
                        [ 0.07202179,  0.93377948,  0.45508945, -0.06685078],
                        [ 0.20099468,  0.66826397,  0.18502794,  0.42627597],
                        [ 2.2287078 ,  2.53909865,  2.51446775,  2.00517466]]),
             np.array([[1.],
                        [0.],
                        [0.],
                        [1.]]))
        ],
        ids=[
            "Test binary accuracy"
        ]

)
def test_binary_accuracy(tr_X, tr_y, test_X, test_y):
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

    ff = FeedForward(net=net)
    bp = BackProp(forward_pass=ff, learning_rate=0.001, loss_function="binary_cross_entropy")
    epoch = 200
    tr = Learner(backprop=bp, epoch=epoch)
    acc = tr.binary_classification_accuracy(test_data=test_X, test_label=test_y)


    assert acc == np.float64(0.5)
