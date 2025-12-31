import numpy as np

def sqrt(x):
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=np.float64)
    mask = x >= 0
    result[mask] = np.sqrt(x[mask])
    return result