# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
from .gp_extract import gp_extract

def gp_getcomplexity(expr):
    """Returns the expressional complexity of an encoded tree or a list of trees."""
    if isinstance(expr, str):
        return _getcomp(expr)
    elif isinstance(expr, list):
        if len(expr) < 1:
            raise ValueError('List must contain at least one valid symbolic expression')
        else:
            comp = sum(_getcomp(e) for e in expr)
            return comp
    else:
        raise ValueError('Illegal argument')

def _getcomp(expr):
    """Get complexity from a single tree."""
    leafcount = expr.count('x') + expr.count('[') + expr.count('z')
    
    ind = [i for i, char in enumerate(expr) if char == '(']
    if not ind:
        return leafcount

    ind = [i - 1 for i in ind]
    comp = 0
    for i in ind:
        _, subtree = gp_extract(i, expr)
        comp += _getnn(subtree)

    comp += leafcount
    return comp

def _getnn(expr):
    """Get number of nodes from a single tree."""
    num_open = expr.count('(')
    num_const = expr.count('[')
    num_inps = expr.count('x') + expr.count('z')
    numnodes = num_open + num_const + num_inps
    return numnodes

