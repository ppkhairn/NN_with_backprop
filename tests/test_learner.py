import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.core.backprop import BackProp
from src.core.layers import NeuNet
from src.core.feed_forward import FeedForward
from src.core.learner import Learner
from src.utils.loss_functions import binary_cross_entropy, mean_squared_error
from src.utils.data_randomize import df_random

# resolve paths
curr_script_path = Path(__file__)
curr_folder_path = curr_script_path.parent
root_folder_path = curr_folder_path.parent
data_folder = root_folder_path / "data/data_to_test"

rand_df = df_random(pd.read_csv(str(data_folder)+"/binary_classification_data.csv"))

@pytest.mark.parametrize(
    ("tr_data"),
    [
        rand_df.head(6)
    ],
    ids=[
        "Test learner"
    ]
)
def test_learner(tr_data):

    tr_X = tr_data.iloc[:, 1:3]
    tr_X = np.array(tr_X)
    tr_y = tr_data.iloc[:, -1]
    tr_y = np.array(tr_y)
    tr_y = tr_y.reshape(tr_y.shape[0], 1)

    net = NeuNet(tr_X, tr_y)
    net.input_layer()
    net.add_hidden_layer(1, "sigmoid")
    net.output_layer(1, "sigmoid")
    net.initialize_weights()

    ff = FeedForward(net)
    bp = BackProp(forward_pass=ff, learning_rate=0.1, loss_function="binary_cross_entropy")
    epoch = 5
    tr = Learner(backprop=bp, epoch=epoch)
    loss = tr.train()

    assert len(loss) == epoch