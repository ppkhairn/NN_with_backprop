import numpy as np
import pandas as pd
from pathlib import Path

curr_script_path = Path(__file__)
curr_folder_path = curr_script_path.parent
data_folder = curr_folder_path.parent
data_content_folder = data_folder / "data_to_test"
# print(f"data_content_folder - {data_content_folder}")


np.random.seed(0)
n_samples = 100

# Class 0: centered around (0,0)
X0 = np.random.randn(2, n_samples) * 0.5 + np.array([[0], [0]])
y0 = np.zeros((1, n_samples))

# Class 1: centered around (2,2)
X1 = np.random.randn(2, n_samples) * 0.5 + np.array([[2], [2]])
y1 = np.ones((1, n_samples))

# Combine
tr_X = np.concatenate([X0, X1], axis=1)  # shape (2, 200)
tr_y = np.concatenate([y0, y1], axis=1)  # shape (1, 200)

df_X = pd.DataFrame(tr_X.T, columns=["x_0", "x_1"])
df_y = pd.DataFrame(tr_y.T, columns=["y"])
df = pd.concat([df_X, df_y], axis=1)
# print(df)

df.to_csv(str(data_content_folder)+"/binary_classification_data.csv")