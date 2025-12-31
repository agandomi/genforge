import numpy as np
def unpack_if_tuple(x):
    """Extract value and penalty — or assign zero-penalty if x is raw array."""
    if isinstance(x, tuple):
        return x[0], x[1]
    else:
        return x, np.zeros_like(x)
    
def divide(a, b):
    a_val, a_pen = unpack_if_tuple(a)
    b_val, b_pen = unpack_if_tuple(b)
    # Safe divide using numpy with error masking
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(a_val, b_val)
        penalty_mask = ~np.isfinite(result)  # True where inf or nan
        result[penalty_mask] = 0  # Replace invalid results with 0
        penalty_array = penalty_mask.astype(float)  # Convert to 1.0 (penalized), 0.0 (safe)
    return result, penalty_array