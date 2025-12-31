# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
def mergegp(gp1, gp2, multirun=False):
    """Merge two GP population structures into a new one.

    Args:
        gp1 (dict): First GP population structure.
        gp2 (dict): Second GP population structure.
        multirun (bool, optional): Flag for multiple runs. Defaults to False.

    Returns:
        dict: Merged GP population structure.
    """
    print("Merging GP populations ...")

    # Check for compatibility between gp1 and gp2
    if gp1['nodes']['inputs'] != gp2['nodes']['inputs']:
        raise ValueError("GP structures cannot be merged because the input nodes are not configured identically.")
    
    if gp1['nodes']['functions'] != gp2['nodes']['functions']:
        raise ValueError("GP structures cannot be merged because the function nodes are not configured identically.")

    # Update the appropriate fields of gp1
    popsize1 = gp1['runcontrol']['pop_size']
    popsize2 = gp2['runcontrol']['pop_size']
    inds = list(range(popsize1, popsize1 + popsize2))

    gp1['pop'].extend(gp2['pop'])
    gp1['fitness']['returnvalues'].extend(gp2['fitness']['returnvalues'])
    gp1['fitness']['complexity'].extend(gp2['fitness']['complexity'])
    gp1['fitness']['nodecount'].extend(gp2['fitness']['nodecount'])
    gp1['fitness']['values'].extend(gp2['fitness']['values'])
    gp1['runcontrol']['pop_size'] = popsize1 + popsize2

    # Check for existence of r2train, r2val, and r2test fields before merging
    if 'r2train' in gp1['fitness'] and 'r2train' in gp2['fitness']:
        gp1['fitness']['r2train'].extend(gp2['fitness']['r2train'])

    if 'r2val' in gp1['fitness'] and 'r2val' in gp2['fitness']:
        gp1['fitness']['r2val'].extend(gp2['fitness']['r2val'])

    if 'r2test' in gp1['fitness'] and 'r2test' in gp2['fitness']:
        gp1['fitness']['r2test'].extend(gp2['fitness']['r2test'])

    # Check for 'best' on validation and test data
    val_present = 'valbest' in gp1['results'] and 'valbest' in gp2['results']
    test_present = 'testbest' in gp1['results'] and 'testbest' in gp2['results']

    # Get complexity values for best and valbest
    if gp1['fitness']['complexityMeasure']:
        traincomp1 = gp1['results']['best']['complexity']
        traincomp2 = gp2['results']['best']['complexity']

        if val_present:
            valcomp1 = gp1['results']['valbest']['complexity']
            valcomp2 = gp2['results']['valbest']['complexity']

        if test_present:
            testcomp1 = gp1['results']['testbest']['complexity']
            testcomp2 = gp2['results']['testbest']['complexity']

    else:
        traincomp1 = gp1['results']['best']['nodecount']
        traincomp2 = gp2['results']['best']['nodecount']

        if val_present:
            valcomp1 = gp1['results']['valbest']['nodecount']
            valcomp2 = gp2['results']['valbest']['nodecount']

        if test_present:
            testcomp1 = gp1['results']['testbest']['nodecount']
            testcomp2 = gp2['results']['testbest']['nodecount']

    if gp1['fitness']['minimisation']:
        if gp2['results']['best']['fitness'] < gp1['results']['best']['fitness'] or \
           (gp2['results']['best']['fitness'] == gp1['results']['best']['fitness'] and traincomp2 < traincomp1):
            gp1['results']['best'] = gp2['results']['best']

        if val_present:
            if gp2['results']['valbest']['valfitness'] < gp1['results']['valbest']['valfitness'] or \
               (gp2['results']['valbest']['valfitness'] == gp1['results']['valbest']['valfitness'] and valcomp2 < valcomp1):
                gp1['results']['valbest'] = gp2['results']['valbest']

        if test_present:
            if gp2['results']['testbest']['testfitness'] < gp1['results']['testbest']['testfitness'] or \
               (gp2['results']['testbest']['testfitness'] == gp1['results']['testbest']['testfitness'] and testcomp2 < testcomp1):
                gp1['results']['testbest'] = gp2['results']['testbest']

    else:  # maximisation
        if gp2['results']['best']['fitness'] > gp1['results']['best']['fitness'] or \
           (gp2['results']['best']['fitness'] == gp1['results']['best']['fitness'] and traincomp2 < traincomp1):
            gp1['results']['best'] = gp2['results']['best']

        if val_present:
            if gp2['results']['valbest']['valfitness'] > gp1['results']['valbest']['valfitness'] or \
               (gp2['results']['valbest']['valfitness'] == gp1['results']['valbest']['valfitness'] and valcomp2 < valcomp1):
                gp1['results']['valbest'] = gp2['results']['valbest']

        if test_present:
            if gp2['results']['testbest']['testfitness'] > gp1['results']['testbest']['testfitness'] or \
               (gp2['results']['testbest']['testfitness'] == gp1['results']['testbest']['testfitness'] and testcomp2 < testcomp1):
                gp1['results']['testbest'] = gp2['results']['testbest']

    # Merge info
    if not gp1['info'].get('merged', False) and not gp2['info'].get('merged', False):
        gp1['info']['mergedPopSizes'] = [popsize1, popsize2]
    elif gp1['info'].get('merged', False) and not gp2['info'].get('merged', False):
        gp1['info']['mergedPopSizes'].append(popsize2)
    elif not gp1['info'].get('merged', False) and gp2['info'].get('merged', False):
        gp1['info']['mergedPopSizes'] = [popsize1] + gp2['info']['mergedPopSizes']
    elif gp1['info'].get('merged', False) and gp2['info'].get('merged', False):
        gp1['info']['mergedPopSizes'] += gp2['info']['mergedPopSizes']

    # Counter for number of times gp1 has experienced a merge
    gp1['info']['merged'] = gp1['info'].get('merged', 0) + 1
    gp1['info']['duplicatesRemoved'] = False

    # Set stop time to time taken for all runs in multirun
    if multirun:
        gp1['info']['stopTime'] = gp2['info']['stopTime']

    return gp1
