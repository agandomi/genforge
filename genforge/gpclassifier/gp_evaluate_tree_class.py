# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np

def gp_evaluate_tree(gene_str, x_data, function_map):
    """
    Evaluates a single gene string.

    Args:
    gene_str (str): The gene expression string (e.g., 'minus(times(x10,x1),plus3(x1001,[-1.34423],x55))').
    x_data (np.array): A numpy array containing input data (e.g., x_data[:, 9] for x10).
    function_map (dict): A dictionary mapping function names to their corresponding Python functions.

    Returns:
    np.array: The result of evaluating the gene string with shape[0] equal to x_data.shape[0].
    """
    def parse_expression(expr):
        """Recursively parses and evaluates the expression."""
        expr = expr.strip()
        
        if expr.startswith('x'):  # It's an input variable like x10
            index = int(expr[1:]) - 1  # Convert 'x10' to index 9
            return x_data[:, index]
        
        elif expr.startswith('[') and expr.endswith(']'):  # It's a constant
            const_value = float(expr.strip('[]'))
            return np.full(x_data.shape[0], const_value)  # Return a constant array
        
        elif '(' in expr and expr.endswith(')'):  # It's a function call
            # Extract the function name and its arguments
            func_name_end = expr.find('(')
            func_name = expr[:func_name_end]
            args_str = expr[func_name_end + 1:-1]  # Remove the outer parentheses
            args = split_args(args_str)
            evaluated_args = [parse_expression(arg) for arg in args]
            return function_map[func_name](*evaluated_args)
        
        else:
            print(gene_str)
            raise ValueError(f"Unexpected expression format: {expr}")

    def split_args(args_str):
        """Splits a comma-separated list of arguments, accounting for nested parentheses."""
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

# Example usage:
# Assuming x_data is a numpy array with shape (100, 10)
# Assuming function_map is a dictionary like {'minus': minus_function, 'times': times_function, 'plus3': plus3_function}
# Example: result = gp_evaluate_tree('minus(times(x10,x1),plus3(x1001,[-1.34423],x55))', x_data, function_map)
