# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np

def gp_evaluate_tree(gene_str, x_data, z_data, function_map):
    """
    Evaluates a single gene string using both x_data (features) and z_data (injected expressions).

    Args:
        gene_str (str): The gene expression string (e.g., 'plus(times(x10,z1),[-1.5])').
        x_data (np.ndarray): Input feature matrix (N, D).
        z_data (np.ndarray): Injected symbolic features (N, M).
        function_map (dict): Mapping of function names to callables returning (value, penalty).

    Returns:
        np.ndarray: Expression values (N,), np.ndarray: Penalty values (N,)
    """

    def parse_expression(expr):
        expr = expr.strip()

        if expr.startswith('x'):
            index = int(expr[1:]) - 1
            return x_data[:, index], np.zeros(x_data.shape[0])

        elif expr.startswith('z'):
            index = int(expr[1:]) - 1
            return z_data[:, index], np.zeros(z_data.shape[0])

        elif expr.startswith('[') and expr.endswith(']'):
            const_value = float(expr.strip('[]'))
            return np.full(x_data.shape[0], const_value), np.zeros(x_data.shape[0])

        elif '(' in expr and expr.endswith(')'):
            func_name_end = expr.find('(')
            func_name = expr[:func_name_end]
            args_str = expr[func_name_end + 1:-1]
            args = split_args(args_str)

            evaluated = [parse_expression(arg) for arg in args]
            values = [v for v, _ in evaluated]
            penalties = [p for _, p in evaluated]

            result_val, result_penalty = function_map[func_name](*values)
            total_penalty = result_penalty.copy()
            for p in penalties:
                total_penalty += p
            return result_val, total_penalty

        else:
            print(f"Malformed expression: {expr}")
            raise ValueError(f"Unexpected expression format: {expr}")

    def split_args(args_str):
        """Splits function arguments, considering nested expressions."""
        args = []
        bracket_level = 0
        current_arg = []
        for char in args_str:
            if char == ',' and bracket_level == 0:
                args.append(''.join(current_arg).strip())
                current_arg = []
            else:
                if char == '(':
                    bracket_level += 1
                elif char == ')':
                    bracket_level -= 1
                current_arg.append(char)
        args.append(''.join(current_arg).strip())
        return args

    return parse_expression(gene_str)