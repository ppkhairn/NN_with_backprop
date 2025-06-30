import numpy as np
import math

def mean_squared_error(y_pred: np.ndarray, y_actual: np.ndarray) -> float:
    loss = 0.5 * np.mean((y_pred - y_actual) ** 2)
    return loss

def binary_cross_entropy(y_pred: np.ndarray, y_actual: np.ndarray) -> float:
    epsilon = 1e-15 #to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # to prevent values like 0 and 1
    loss = -np.mean((y_actual * np.log(y_pred)) + ((1 - y_actual)*np.log(1 - y_pred)))
    return loss

