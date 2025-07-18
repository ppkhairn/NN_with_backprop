import numpy as np
from src.nn_with_backprop_pk.utils.activations import sigmoid, tanh

def diff_mse(y_pred: np.ndarray, y_actual: np.ndarray) -> np.ndarray:
    return y_pred - y_actual

def diff_binary_cross_entropy(y_pred: np.ndarray, y_actual: np.ndarray) -> np.ndarray:
    epsilon = 1e-15 #to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # to prevent values like 0 and 1
    return -((y_actual / y_pred) - ((1-y_actual)/(1-y_pred)))

# def diff_sigmoid(x: np.ndarray) -> np.ndarray:
#     s = sigmoid(x)
#     return s * (1 -s)

def diff_sigmoid(x: np.ndarray) -> np.ndarray:
    # s = sigmoid(x)
    return x * (1 -x)

def diff_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

def diff_tanh(x: np.ndarray) -> np.ndarray:
    t = tanh(x)
    return 1 - t**2