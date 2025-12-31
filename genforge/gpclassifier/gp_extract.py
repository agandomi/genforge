# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
def gp_extract(index, parentExpr):
    """
    Extract a subtree from an encoded tree expression.

    Args:
    index (int): The index in parentExpr of the root node of the subtree to be extracted.
    parentExpr (str): The parent string expression.

    Returns:
    tuple: (mainTree, subTree)
           - mainTree: parentExpr with the removed subtree replaced by '$'.
           - subTree: The extracted subtree.
    """
    cnode = parentExpr[index]
    iplus = index + 1
    iminus = index - 1
    endpos = len(parentExpr)

    if cnode == 'x':  # Extracting an input terminal (e.g., x1, x2, etc.)
        section = parentExpr[iplus:endpos]
        inp_comma_ind = section.find(',')
        inp_brack_ind = section.find(')')

        # If none found then string must consist of a single input
        if inp_brack_ind == -1 and inp_comma_ind == -1:
            mainTree = '$'
            subTree = parentExpr
        else:
            inp_ind = sorted([i for i in [inp_brack_ind, inp_comma_ind] if i != -1])
            final_ind = inp_ind[0] + index + 1
            subTree = parentExpr[index:final_ind]
            mainTree = parentExpr[:iminus] + '$' + parentExpr[final_ind:endpos]
        return mainTree, subTree

    elif cnode == '[':  # Extracting an Ephemeral Random Constant (ERC)
        cl_sbr = parentExpr[iplus:endpos].find(']')
        final_ind = cl_sbr + index + 1
        subTree = parentExpr[index:final_ind]
        mainTree = parentExpr[:iminus] + '$' + parentExpr[final_ind + 1:endpos]
        return mainTree, subTree

    elif cnode == '?':  # ERC token
        subTree = cnode
        mainTree = parentExpr[:index] + '$' + parentExpr[index + 1:]
        return mainTree, subTree

    else:  # Otherwise extract a tree with a function node as root
        search_seg = parentExpr[index:endpos]

        # Get indices of open and close brackets
        op = [i for i, ch in enumerate(search_seg) if ch == '(']
        cl = [i for i, ch in enumerate(search_seg) if ch == ')']

        # Compare indices to determine the point where num_open = num_closed
        tr_op = op[1:]  # Skip the first open bracket
        l_tr_op = len(tr_op)

        hibvec = [tr_op[i] - cl[i] for i in range(l_tr_op)]

        cl_ind = next((i for i, val in enumerate(hibvec) if val > 0), None)

        if cl_ind is None:
            j = cl[len(op) - 1]
        else:
            j = cl[cl_ind]

        subTree = search_seg[:j + 1]
        mainTree = parentExpr[:iminus] + '$' + parentExpr[j + index + 1:endpos]
        return mainTree, subTree
