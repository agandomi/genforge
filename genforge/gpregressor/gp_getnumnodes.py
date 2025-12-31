# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
def gp_getnumnodes(expr):
    """Returns the number of nodes in an encoded tree expression or the total node count for a list of expressions."""
    if isinstance(expr, str):
        return _getnn(expr)
    elif isinstance(expr, list):
        if len(expr) < 1:
            raise ValueError('List must contain at least one valid symbolic expression')
        else:
            numnodes = sum(_getnn(e) for e in expr)
            return numnodes
    else:
        raise ValueError('Illegal argument')

def _getnn(expr):
    """Get number of nodes from a single symbolic string."""
    num_open = expr.count('(')
    num_const = expr.count('[')
    num_inps = expr.count('x') + expr.count('z')
    numnodes = num_open + num_const + num_inps
    return numnodes
