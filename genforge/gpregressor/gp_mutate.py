# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
from .gp_treegen import gp_treegen
from .gp_picknode import gp_picknode

def gp_mutate(gp, expr, id_pop=0):
    """Mutate an encoded symbolic tree expression."""
    rng = gp.config['runcontrol']['RNG']
    mutate_par_cumsum = gp.config['operator']['mutate_par_cumsum'][id_pop]
    max_mutate_depth = gp.config['tree']['max_mutate_depth'][id_pop]
    mutate_gaussian_std = gp.config['operator']['mutate_gaussian_std'][id_pop]
    const_range = gp.config['nodes']['const']['range'][id_pop]
    num_dec_places = gp.config['nodes']['const']['num_dec_places'][id_pop]
    rnum = rng.random()

    while True:  # Loop until a mutation occurs
        # Ordinary subtree mutation
        if rnum <= mutate_par_cumsum[0]:
            # First get the string position of the node
            position = gp_picknode(gp, expr, 0, id_pop)
            # Remove the logical subtree for the selected node
            # and return the main tree with '$' replacing the extracted subtree
            maintree = expr[:position[1]] + '$' + expr[position[2]+1:]
            # Generate a new tree to replace the old
            # Pick a depth between 1 and gp.treedef.max_mutate_depth;
            depth = rng.integers(1, max_mutate_depth + 1)
            new_tree = gp_treegen(gp, id_pop, depth)
            # Replace the '$' with the new tree
            expr_mut = maintree.replace('$', new_tree)
            break

        # Shuffle inputs
        elif rnum <= mutate_par_cumsum[1]:
            # Just input nodes
            position = gp_picknode(gp, expr, 2, id_pop)
            if position == -1:
                rnum = 0
                continue
            else:
                maintree = expr[:position[1]] + '$' + expr[position[2]+1:]
                terminal = f"x{rng.choice(gp.userdata['pop_idx'][id_pop])+1}"
                expr_mut = maintree.replace('$', terminal)
                break

        # Constant Gaussian perturbation
        elif rnum <= mutate_par_cumsum[2]:
            # Just Constant nodes
            position = gp_picknode(gp, expr, 3, id_pop)
            if position == -1:
                rnum = 0
                continue
            else:
                maintree = expr[:position[1]] + '$' + expr[position[2]+1:]
                subtree = position[0]
                # Process constant
                oldconst = float(subtree[1:-1])
                newconst = oldconst + (rng.standard_normal() * mutate_gaussian_std)
                # Use appropriate string formatting for new constant
                if newconst == int(newconst):
                    new_tree = f"[{int(newconst)}]"
                else:
                    new_tree = f"[{newconst:.{num_dec_places}f}]"  # Adjust format as needed
                # Replace '$' with the new tree
                expr_mut = maintree.replace('$', new_tree)
                break

        # Make constant zero
        elif rnum <= mutate_par_cumsum[3]:
            position = gp_picknode(gp, expr, 3, id_pop)
            if position == -1:
                rnum = 0
                continue
            else:
                maintree = expr[:position[1]] + '$' + expr[position[2]+1:]
                expr_mut = maintree.replace('$', '[0]')  # Replace it with zero
                break

        # Randomize constant
        elif rnum <= mutate_par_cumsum[4]:
            position = gp_picknode(gp, expr, 3, id_pop)
            if position == -1:
                rnum = 0
                continue
            else:
                maintree = expr[:position[1]] + '$' + expr[position[2]+1:]
                # Generate a constant in the range
                const = rng.random() * (const_range[1] - const_range[0]) + const_range[0]
                # Convert the constant to square bracketed string form
                if const == int(const):
                    arg = f"[{int(const)}]"
                else:
                    arg = f"[{const:.{num_dec_places}f}]"  # Adjust format as needed
                expr_mut = maintree.replace('$', arg)  # Insert new constant
                break

        # Set a constant to 1
        elif rnum <= mutate_par_cumsum[5]:
            position = gp_picknode(gp, expr, 3, id_pop)
            if position == -1:
                rnum = 0
                continue
            else:
                maintree = expr[:position[1]] + '$' + expr[position[2]+1:]
                expr_mut = maintree.replace('$', '[1]')  # Replace it with one
                break

    return expr_mut
