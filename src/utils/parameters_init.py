import numpy as np

def xavier_init(shape):
    if not isinstance(shape, tuple):
        raise TypeError(f"Shape must be a tuple, got {type(shape)}: {shape}")
    if len(shape) == 1:
        dim = shape[0]
        limit = np.sqrt(6 / (dim + dim))
    elif len(shape) == 2:
        in_dim, out_dim = shape
        limit = np.sqrt(6 / (in_dim + out_dim))
    else:
        raise ValueError(f"Unsupported Shape: {shape}")
    
    return np.random.uniform(-limit, limit, size=shape)

def he_init(shape):
    if not isinstance(shape, tuple):
        raise TypeError(f"Shape must be a tuple, got {type(shape)}: {shape}")
    if len(shape) == 1:
        in_dim = shape[0]
    elif len(shape) == 2:
        in_dim, _ = shape
    else:
        raise ValueError(f"Unsupported Shape: {shape}")
    std = np.sqrt(2 / in_dim)

    return np.random.randn(*shape) * std