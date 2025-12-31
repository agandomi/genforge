# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
from .scangenes import scangenes
import os

def gpmodelvars(gp, ID):
    """Display the frequency of input variables present in the specified model."""
    global runname, fldres
    hitvec = []

    if isinstance(ID, int):
        if ID < 1 or ID > len(gp.population):
            print('Invalid value of supplied numerical identifier ID.')
            return

        model = gp.population[ID - 1]
        title_str = f'Input frequency in individual with ID: {ID}'
    
    elif isinstance(ID, str) and ID.lower() == 'best':
        model = gp.results.best.individual
        title_str = 'Input frequency in best individual.'
    
    elif isinstance(ID, str) and ID.lower() == 'valbest':
        if 'valbest' not in gp.results:
            raise ValueError('No validation data was found. Try GPMODELVARS(GP, "BEST") instead.')
        model = gp.results.valbest.individual
        title_str = 'Input frequency in best validation individual.'
    
    elif isinstance(ID, str) and ID.lower() == 'testbest':
        if 'testbest' not in gp.results:
            raise ValueError('No test data was found. Try GPMODELVARS(GP, "BEST") instead.')
        model = gp.results.testbest.individual
        title_str = 'Input frequency in best test individual.'
    
    elif isinstance(ID, list):
        model = ID
        title_str = 'Input frequency in user model.'
    
    elif isinstance(ID, dict) and 'genes' in ID:
        model = ID['genes']['geneStrs']
        title_str = 'Input frequency in user model.'
    
    else:
        raise ValueError('Illegal argument')

    numx = []
    if isinstance(gp.userdata.xindex, list):
        for i in gp.userdata.xindex:
            numx = list(set(numx) | set(i))
    else:
        numx = gp.userdata.xindex

    hitvec = scangenes(model, len(numx))

    if not isinstance(ID, int):  # Suppress graphical output for non-integer IDs
        return hitvec

    # Plot results as bar chart
    import matplotlib.pyplot as plt
    plt.bar(range(len(hitvec)), hitvec, color=[0, 0.45, 0.74])
    plt.xlabel('Input')
    plt.ylabel('Input frequency')
    plt.title(title_str)
    plt.xticks(range(len(numx)), [f'x{i}' for i in numx])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fldres, runname, 'Input_Frequency.png'))
    plt.show()

    return hitvec
