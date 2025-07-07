import pytest
import pandas as pd
from src.utils.data_split import train_test_split
from pathlib import Path
from math import ceil

# resolve paths
curr_script = Path(__file__)
curr_folder = curr_script.parent
root_folder = curr_folder.parent
data_contents = root_folder / "data/data_to_test"

@pytest.mark.parametrize(
    ("df", "split_factor"),
    [
        (pd.read_csv(data_contents / "binary_classification_data_4_f.csv"), 0.7),
        (pd.read_csv(data_contents / "binary_classification_data_4_f.csv"), 0.4)
    ],
    ids=[
        "Test data split within split factor limits", "Test data split with oob split factor limits"
    ]
)

def test_data_split(df, split_factor):

    if 0.6 <= split_factor <= 0.9:
        tr_data, test_data = train_test_split(df, split_factor)
        assert tr_data.shape[0] == ceil(df.shape[0]*split_factor)
        assert tr_data.shape[1] == df.shape[1]
        assert test_data.shape[0] == df.shape[0] - ceil(df.shape[0]*split_factor)
        assert test_data.shape[1] == df.shape[1]
    else:
        with pytest.raises(ValueError):
            tr_data, test_data = train_test_split(df, split_factor)