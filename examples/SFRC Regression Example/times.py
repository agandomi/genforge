import numpy as np
def unpack_if_tuple(x):
    """Extract value and penalty — or assign zero-penalty if x is raw array."""
    if isinstance(x, tuple):
        return x[0], x[1]
    else:
        return x, np.zeros_like(x)
    
def times(a,b):
    a_val, a_pen = unpack_if_tuple(a)
    b_val, b_pen = unpack_if_tuple(b)
    
    return a_val*b_val, np.zeros_like(a_val)