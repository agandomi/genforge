# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
def gp_extract(index, parentExpr):
    """
    Extract a subtree from an encoded tree expression.

    Args:
        index (int): Index of the starting character (e.g., 'x', 'z', '[', or function) to extract from.
        parentExpr (str): Full expression string.

    Returns:
        tuple: (mainTree, subTree)
            - mainTree: parentExpr with the extracted subexpression replaced by '$'
            - subTree: The extracted subexpression
    """
    cnode = parentExpr[index]
    iplus = index + 1
    iminus = max(0, index - 1)
    endpos = len(parentExpr)

    # Handle x or z terminals
    if cnode in ('x', 'z'):
        section = parentExpr[iplus:endpos]
        comma_ind = section.find(',')
        close_ind = section.find(')')

        # Find where this terminal ends
        if comma_ind == -1 and close_ind == -1:
            return '$', parentExpr
        else:
            terminal_end = min(
                [i for i in [comma_ind, close_ind] if i != -1]
            ) + index + 1
            subTree = parentExpr[index:terminal_end]
            mainTree = parentExpr[:index] + '$' + parentExpr[terminal_end:]
            return mainTree, subTree

    # Handle constant terminal like [1.23]
    elif cnode == '[':
        cl_sbr = parentExpr[iplus:endpos].find(']')
        final_ind = cl_sbr + index + 1
        subTree = parentExpr[index:final_ind]
        mainTree = parentExpr[:index] + '$' + parentExpr[final_ind + 1:]
        return mainTree, subTree

    # Handle constant token '?'
    elif cnode == '?':
        subTree = cnode
        mainTree = parentExpr[:index] + '$' + parentExpr[index + 1:]
        return mainTree, subTree

    # Function root node
    else:
        search_seg = parentExpr[index:endpos]
        num_open = 0
        for j, char in enumerate(search_seg):
            if char == '(':
                num_open += 1
            elif char == ')':
                num_open -= 1
                if num_open == 0:
                    subTree = search_seg[:j + 1]
                    mainTree = parentExpr[:index] + '$' + parentExpr[index + j + 1:]
                    return mainTree, subTree

        raise ValueError(f"Malformed expression: {parentExpr} — unable to extract function node.")

