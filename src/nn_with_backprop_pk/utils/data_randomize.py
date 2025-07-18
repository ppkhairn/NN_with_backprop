import random
import pandas as pd

def df_random(tr_data: pd.DataFrame) -> pd.DataFrame:

    if not isinstance(tr_data, pd.DataFrame):
        raise NotImplementedError("Code not yet implemented for data other than pandas dataframe!")

    random_list = random.sample(range(0, tr_data.shape[0]), tr_data.shape[0])
    rand_df = tr_data.loc[random_list]

    return rand_df
