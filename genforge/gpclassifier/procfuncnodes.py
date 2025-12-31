# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import inspect

def procfuncnodes(gp):
    """Process required function node information prior to a run."""
    # Loop through function nodes and generate arity list
    for i in range(len(gp['nodes']['functions']['name'])):
        func_name = gp['nodes']['functions']['name'][i]
        arity = len(inspect.signature(func_name).parameters)
        
        # Some functions have a variable number of input arguments (e.g., random functions)
        # In this case, generate an error message and exit
        if arity == -1:
            raise ValueError(f"The function {func_name} may not be used (directly) as a function node because it has a variable number of arguments.")
        
        gp['nodes']['functions']['arity'][i] = arity
    
    if 'active' not in gp['nodes']['functions'] or not gp['nodes']['functions']['active']:
        gp['nodes']['functions']['active'] = [1] * len(gp['nodes']['functions']['name'])
    
    gp['nodes']['functions']['active'] = [bool(x) for x in gp['nodes']['functions']['active']]
    
    # Check max number of allowed functions not exceeded
    gp['nodes']['functions']['num_active'] = sum(gp['nodes']['functions']['active'])
    if gp['nodes']['functions']['num_active'] > 22:
        raise ValueError("Maximum number of active functions allowed is 22.")
    
    # Generate single char Active Function Identifiers (afid) (a->z excluding x, e, i, j)
    # Exclusions are because 'x' is reserved for input nodes, 'e' is used for expressing numbers in standard form by Python,
    # and by default, 'i' and 'j' represent sqrt(-1).
    charnum = 96
    skip = 0
    afid = []
    for i in range(gp['nodes']['functions']['num_active']):
        while True:
            if (charnum + i + skip) in [101, 105, 106, 120]:  # ASCII values for 'e', 'i', 'j', 'x'
                skip += 1
            else:
                break
        afid.append(chr(charnum + i + skip))
    
    # Extract upper case active function names for later use
    gp['nodes']['functions']['afid'] = afid
    temp = [None] * gp['nodes']['functions']['num_active']
    
    if len(gp['nodes']['functions']['name']) != len(gp['nodes']['functions']['active']):
        raise ValueError("There must be the same number of entries in gp.nodes.functions.name and gp.nodes.functions.active. Check your config file.")
    
    active_names = [gp['nodes']['functions']['name'][i] for i, active in enumerate(gp['nodes']['functions']['active']) if active]
    gp['nodes']['functions']['active_name_UC'] = [name.upper() for name in active_names]
    
    # Generate index locators for arity > 0 and arity == 0 active functions
    active_ar = [gp['nodes']['functions']['arity'][i] for i, active in enumerate(gp['nodes']['functions']['active']) if active]
    fun_argt0 = [arity > 0 for arity in active_ar]
    fun_areq0 = [not arg for arg in fun_argt0]
    
    gp['nodes']['functions']['afid_argt0'] = [afid[i] for i, arg in enumerate(fun_argt0) if arg]
    gp['nodes']['functions']['afid_areq0'] = [afid[i] for i, arg in enumerate(fun_areq0) if arg]
    gp['nodes']['functions']['arity_argt0'] = [arity for arity in active_ar if arity > 0]
    
    gp['nodes']['functions']['fun_lengthargt0'] = len(gp['nodes']['functions']['afid_argt0'])
    gp['nodes']['functions']['fun_lengthareq0'] = len(gp['nodes']['functions']['afid_areq0'])
    
    return gp
