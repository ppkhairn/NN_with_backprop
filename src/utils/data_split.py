from typing import Tuple
import pandas as pd
from math import ceil

class SplitFactor(float):
    def __new__(cls, val):
        if not 0.6 <= val <=0.9:
            raise ValueError(f"The value must be between 0.6 and 0.9. Provided {val}.")
        return float.__new__(cls, val)

def train_test_split(df: pd.DataFrame, split_factor: SplitFactor) -> Tuple[pd.DataFrame, pd.DataFrame]:

    split_factor = SplitFactor(split_factor)
    total_rows = df.shape[0]
    ratio = ceil(total_rows * split_factor)
    train_data = df.iloc[:ratio, :]
    test_data = df.iloc[ratio:, :] 

    return train_data, test_data