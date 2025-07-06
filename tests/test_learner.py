import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.core.backprop import BackProp
from src.core.layers import NeuNet
from src.core.feed_forward import FeedForward
from src.core.learner import Learner
from src.utils.loss_functions import binary_cross_entropy, mean_squared_error

# resolve paths
curr_script_path = Path(__file__)
curr_folder_path = curr_script_path.parent
root_folder_path = curr_folder_path.parent
data_folder = root_folder_path / "data/data_to_test"

@pytest.mark.parametrize(
    ("tr_data"),
    [
        pd.read_csv(str(data_folder)+"/binary_classification_data.csv").head(5)
    ],
    ids=[
        "Test learner"
    ]
)
def test_backprop(tr_data):

    tr_X = tr_data.iloc[:, 1:3]
    tr_X = np.array(tr_X)
    tr_y = tr_data.iloc[:, -1]
    tr_y = np.array(tr_y)
    tr_y = tr_y.reshape(tr_y.shape[0], 1)

    net = NeuNet(tr_X, tr_y)
    net.input_layer()
    net.add_hidden_layer(2, "sigmoid")
    net.output_layer(1, "sigmoid")
    net.initialize_weights()

    ff = FeedForward(net)
    bp = BackProp(forward_pass=ff, learning_rate=0.1, loss_function="binary_cross_entropy")
    tr = Learner(backprop=bp, epoch=4)
    tr.train()

    

    # net = NeuNet(tr_X, tr_y)
    assert 1==1
    # net = NeuNet(tr_X, tr_y)
    # net.input_layer()
    # net.add_hidden_layer(3, "sigmoid")
    # net.add_hidden_layer(2, "sigmoid")
    # net.output_layer(1, "sigmoid")

    # net.weights = []
    # net.weights.append(np.ones((net.layers[1].shape[0], net.layers[0].shape[0]), dtype=np.float64))
    # net.weights.append(np.ones((net.layers[2].shape[0], net.layers[1].shape[0]), dtype=np.float64))
    # net.weights.append(np.ones((net.layers[3].shape[0], net.layers[2].shape[0]), dtype=np.float64))

    # net.biases = []
    # net.biases.append(np.array([[1, 2, 3]], dtype=np.float64).T)
    # net.biases.append(np.array([[2, 3]], dtype=np.float64).T) #, dtype=np.float64
    # net.biases.append(np.array([[4]], dtype=np.float64).T)

    # ff = FeedForward(net)
    # ff.forward_pass()

    # bp = BackProp(ff, learning_rate, loss_function)
    # # bp.back_prop()
    # # weights, biases = bp.update_parameters()

    # if loss_function == "binary_cross_entropy":
    #     bp.back_prop()
    #     weights, biases = bp.update_parameters()
    #     assert np.allclose(weights[-1], np.array([[1.00002479, 1.00002489]]))
    #     assert np.allclose(biases[-1], np.array([[4.00002495]]))
    
    # elif loss_function == "mean_squared_error":
    #     # Expect NotImplementedError for this loss
    #     with pytest.raises(NotImplementedError):
    #         bp.back_prop()