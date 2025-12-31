# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
def scangenes(genes, numx):
    """
    Scan a single multigene individual for all input variables and return a frequency vector.
    
    Args:
    genes (list): A list of genes, where each gene is a list of encoded expressions.
    numx (int): The number of input variables.
    
    Returns:
    list: A frequency vector indicating the number of times each input variable appears.
    """
    numtrees = len(genes)
    xhits = [0] * numx

    for jj in range(numtrees):
        numgenes = len(genes[jj])
        for i in range(numx):
            istr = f'x{i + 1}'  # MATLAB indices are 1-based, Python's are 0-based
            for j in range(numgenes):
                k1 = genes[jj][j].count(f'{istr},')
                k2 = genes[jj][j].count(f'{istr})')

                # Workaround for special case (trees containing a single terminal node)
                if genes[jj][j] == istr:
                    xhits[i] += 1

                xhits[i] += k1 + k2

    return xhits
