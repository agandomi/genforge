# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
import re
import random

def gp_treegen(gp, id_pop=0, fixed_depth=False):
    """Generate a new encoded GP tree expression."""
    
    max_depth = gp.config['tree']['max_depth'][id_pop] if not fixed_depth else fixed_depth
    # if a fixed depth tree is required use 'full' build method
    build_method = gp.config['tree']['build_method'] if not fixed_depth else 1  # 1=full 2=grow 3=ramped 1/2 and 1/2
    p_ERC = gp.config['nodes']['const']['p_ERC'][id_pop]
    p_int = gp.config['nodes']['const']['p_int'][id_pop]
    pop_idx = [findex + 1 for findex in gp.userdata['pop_idx'][id_pop]]
    num_inp = len(gp.userdata['pop_idx'][id_pop])
    range_ = gp.config['nodes']['const']['range'][id_pop]
    rangesize = range_[1] - range_[0]
    fi = gp.config['nodes']['const']['num_dec_places'][id_pop]
    ERC_generated = False
    
    arity = gp.config['nodes']['functions']['arity'][id_pop]
    arity_0 = [index for index, value in enumerate(arity) if value == 0]
    arity_m = [index for index, value in enumerate(arity) if value > 0]
    name = gp.config['nodes']['functions']['name'][id_pop]
    rng = gp.config['runcontrol']['RNG']
    # afid_argt0 = gp.config['nodes']['functions']['afid_argt0']
    # afid_areq0 = gp.config['nodes']['functions']['afid_areq0']
    # arity_argt0 = gp.config['nodes']['functions']['arity_argt0']
    # fun_lengthargt0 = gp.config['nodes']['functions']['fun_lengthargt0']
    # fun_lengthareq0 = gp.config['nodes']['functions']['fun_lengthareq0']
    
    
    # if using ramped 1/2 and 1/2 then randomly choose max_depth and build_method
    max_depth_temp = rng.integers(1, max_depth + 1) if build_method == 3 else max_depth
    # set either to 'full' or 'grow' for duration of function
    build_method_temp = rng.integers(1, 2 + 1) if build_method ==3 else build_method
    
    # initial string structure (nodes/subtrees to be built are marked with the $ token)
    treestr = '($)'
    
    # recurse through branches
    while True:
        # find the first node token $ and store string position
        node_token_ind = treestr.find('$')
        
        # breakout if no more nodes to fill
        if node_token_ind == -1:
            break
        
        # get the first node position and process it
        node_token = node_token_ind
        # replace this node token with 'active' node token, @
        treestr = treestr[:node_token] + '@' + treestr[node_token + 1:]
        
        # count brackets from beginning of string to @ to work out depth of @
        left_seg = treestr[:node_token]
        num_open = left_seg.count('(')
        num_closed = left_seg.count(')')
        depth = num_open - num_closed
        
        # choose type of node to insert based on depth and building method. 
        # If root node then pick a non-terminal node (unless otherwise specified).
        # node_type 1=internal 2=terminal
        if depth == 1:
            node_type = 1 if max_depth_temp != 1 else 2 # but if max_depth is 1 must always pick a terminal
            # if less than max_depth and method is 'full' then only pick a function with arity>0
        elif build_method_temp == 1 and depth < max_depth_temp:
            node_type = 1
        elif depth == max_depth_temp:
            node_type = 2
        else: # randomly pick internal or terminal
            node_type = rng.integers(1, 3)
        
        if node_type == 1:
            fun_choice = rng.choice(arity_m)
            fun_name = name[fun_choice]
            num_fun_args = arity[fun_choice]
            
            fun_rep_str = f"{fun_name}($"
            for j in range(1, num_fun_args):
                fun_rep_str += ",$"
            fun_rep_str += ")"
            
            # replace active node token @ with replacement string
            treestr = treestr.replace('@', fun_rep_str, 1)
        
        elif node_type == 2:
            # choose a function or input with equal probability
            if len(arity_0) > 0 and (rng.random() >= 0.5 or (num_inp == 0 and p_ERC == 0)):
                fun_choice = rng.choice(arity_0)
                fun_name = name[fun_choice]
                # create appropriate replacement string for 0 arity function
                fun_rep_str = f"{fun_name}()"
                # now replace active node token @ with replacement string
                treestr = treestr.replace('@', fun_rep_str, 1)
            else: # choose an input (if it exists) or check if ERCs are switched on, if so pick a ERC node/arithmetic tree token with p_ERC prob.
                term = rng.random() >= p_ERC
                if term: # then use an ordinary terminal not a constant
                    inp_choice = rng.choice(pop_idx)
                    treestr = treestr.replace('@', f"x{inp_choice}", 1)
                else: # use an ERC token (? character)
                    treestr = treestr.replace('@', '?', 1)
                    ERC_generated = True

    # constant processing
    if ERC_generated:
        # find ERC locations
        const_ind = [m.start() for m in re.finditer(r'\?', treestr)]
        num_constants = len(const_ind)
        # generate required number of constants in the allowed range
        const = rng.random(num_constants) * rangesize + range_[0]

        # process integer ERCs if enabled
        if p_int > 0:
            ints = rng.random(num_constants) <= p_int
            const[ints] = np.round(const[ints])
        else:
            ints = [False] * num_constants

        # loop through ERC locations and replace with constant values
        for k in range(num_constants):
            if ints[k]:
                arg = f"[{int(const[k])}]"
            else:
                arg = f"[{const[k]:.{fi}f}]"
            
            # replace token with const
            treestr = treestr.replace('?', arg, 1)

    treestr = treestr[1:-1]
    return treestr
