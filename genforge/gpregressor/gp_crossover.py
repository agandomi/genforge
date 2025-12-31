# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
from .gp_picknode import gp_picknode

def gp_crossover(mum, dad, gp, id_pop):
    """Sub-tree crossover of encoded tree expressions to produce 2 new ones."""
    # Select random crossover nodes in mum and dad expressions
    m_position = gp_picknode(gp, mum, 0, id_pop)
    d_position = gp_picknode(gp, dad, 0, id_pop)

    # Extract main and subtree expressions
    m_main = mum[:m_position[1]] + '$' + mum[m_position[2]+1:]
    d_main = dad[:d_position[1]] + '$' + dad[d_position[2]+1:]

    # Combine to form 2 new GP trees
    daughter = m_main.replace('$', d_position[0])
    son = d_main.replace('$', m_position[0])

    return son, daughter
