# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import random
import numpy as np

def findstring(main_string, substring):
    start_indices = []
    end_indices = []
    start_index = main_string.find(substring)
    if start_index != -1:
        end_indices.append(start_index + len(substring) - 1)
    while start_index != -1:
        start_indices.append(start_index)
        start_index = main_string.find(substring, start_index + 1)
        if start_index != -1:
            end_indices.append(start_index + len(substring) - 1)
    return start_indices, end_indices

def gp_picknode(gp, expression, node_type=0, id_pop=0):
    """
    Pick a random node from the expression tree based on the node type.

    Args:
        gp (object): The GP object with configuration and metadata.
        expression (str): The expression tree as a string.
        node_type (int): Type of node to pick:
                         0 = any node,
                         1 = function node,
                         2 = input node (x or z),
                         3 = constant node.
        id_pop (int): Population index for accessing population-specific config.

    Returns:
        tuple: (node_str, start_index, end_index) or -1 if no valid node found.
    """
    rng = gp.config['runcontrol']['RNG']
    expression2 = expression
    functions = tuple(gp.config['nodes']['functions']['name'][id_pop])

    not_done = True
    id_del = []
    nodes = []
    type_node = []
    expr = expression

    while not_done:
        if expr.startswith(','):
            expr = expr[1:]
            id_comma = expression.find(',')
            expression = expression[:id_comma] + '#' + expression[id_comma + 1:]
            id_del.append(id_comma)

        # Function nodes
        if expr.startswith(functions):
            idx_fun = next((i for i, func in enumerate(functions) if expr.startswith(func)), None)
            num_open = 0
            for j in range(len(functions[idx_fun]), len(expr)):
                if expr[j] == '(':
                    num_open += 1
                elif expr[j] == ')':
                    num_open -= 1
                if num_open == 0:
                    node1 = expr[:j + 1]
                    break
            id_start, id_end = findstring(expression, node1)
            nodes.append((node1, id_start[0], id_end[0]))
            type_node.append(1)
            expression = expression[:id_start[0]] + '#' * len(node1) + expression[id_end[0] + 1:]
            for j in range(id_start[0], id_end[0] + 1):
                id_del.append(j)
            expr = ''.join([char for i, char in enumerate(expression) if i not in id_del])

        # Input terminals: x or z
        elif expr.startswith(('x', 'z')):
            prefix = expr[0]
            num_digit = 0
            for j in range(1, len(expr)):
                if expr[1:j+1].isdigit():
                    num_digit += 1
                else:
                    break
            node1 = expr[:num_digit+1]
            id_start, id_end = findstring(expression, node1)
            nodes.append((node1, id_start[0], id_end[0]))
            type_node.append(2)  # Treat both x and z as input nodes
            expression = expression[:id_start[0]] + '#' * len(node1) + expression[id_end[0] + 1:]
            for j in range(id_start[0], id_end[0] + 1):
                id_del.append(j)
            expr = ''.join([char for i, char in enumerate(expression) if i not in id_del])

        # Constant nodes
        elif expr.startswith('['):
            for j in range(1, len(expr) + 1):
                if expr[:j].endswith(']'):
                    break
            node1 = expr[:j]
            id_start, id_end = findstring(expression, node1)
            nodes.append((node1, id_start[0], id_end[0]))
            type_node.append(3)
            expression = expression[:id_start[0]] + '#' * len(node1) + expression[id_end[0] + 1:]
            for j in range(id_start[0], id_end[0] + 1):
                id_del.append(j)
            expr = ''.join([char for i, char in enumerate(expression) if i not in id_del])

        if len(np.unique(id_del)) == len(expression):
            not_done = False

    # Select a node to return
    if nodes:
        if node_type == 0:
            selected_node = tuple(rng.choice(np.array(nodes,  dtype=object)))
        elif node_type in type_node:
            id_node_type = list(np.where(np.array(type_node) == node_type)[0])
            selected_node = tuple(rng.choice(np.array([nodes[nn] for nn in id_node_type], dtype=object)))
        else:
            return -1  # No node of requested type
        return selected_node
    else:
        return -1  # No nodes found
