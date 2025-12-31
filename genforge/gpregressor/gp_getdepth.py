# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
def gp_getdepth(expr):
    """
    Calculate the depth of an expression based on the number of open and close brackets.

    Args:
    expr (str): The expression as a string.

    Returns:
    int: The depth of the expression.
    """

    # Replace empty parentheses '()' with an empty string
    expr = expr.replace('()', '')

    # Find indices of open and close brackets
    open_br = [i for i, char in enumerate(expr) if char == '(']
    close_br = [i for i, char in enumerate(expr) if char == ')']
    num_open = len(open_br)

    if num_open == 0:  # i.e., a single node
        depth = 1
        return depth
    elif num_open == 1:
        depth = 2
        return depth
    else:
        # depth = max consecutive number of open brackets + 1
        br_vec = [0] * len(expr)
        for i in open_br:
            br_vec[i] = 1
        for i in close_br:
            br_vec[i] = -1
        cumsum_br_vec = [sum(br_vec[:i+1]) for i in range(len(br_vec))]
        depth = max(cumsum_br_vec) + 1
        return depth


