# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np

def gp_individuals_ensemble(gp):
    """Initializes a run."""
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    # num_class = gp.config['runcontrol']['num_class']
    ## Initialize individuals and track attributes
    # Individuals
    gp.individuals = {
        'weight_genes': [[None for _ in range(pop_size)] for _ in range(num_pop)],
        'ensemble_idx': np.zeros((pop_size, num_pop)),
        'ensemble_weight': [None for _ in range(pop_size)],  # Fixed
        'gene_output': {
            'train': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            'validation': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            'test': [[None for _ in range(pop_size)] for _ in range(num_pop)],
        },
        'prob': {
            'isolated': {
                'train': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                'validation': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                'test': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            },
            'ensemble': {
                'train': [None for _ in range(pop_size)],
                'validation': [None for _ in range(pop_size)],
                'test': [None for _ in range(pop_size)],
            },
        },
        'loss': {
            'isolated': {
                'train': np.zeros((pop_size, num_pop)),
                'validation': np.zeros((pop_size, num_pop)),
                'test': np.zeros((pop_size, num_pop)),
            },
            'ensemble': {
                'train': np.zeros((pop_size)),
                'validation': np.zeros((pop_size)),
                'test': np.zeros((pop_size)),
            },
        },
        'yp': {
            'isolated': {
                'train': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                'validation': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                'test': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            },
            'ensemble': {
                'train': [None for _ in range(pop_size)],
                'validation': [None for _ in range(pop_size)],
                'test': [None for _ in range(pop_size)],
            },
        },
        'complexity': {
            'isolated': np.zeros((pop_size, num_pop)),
            'ensemble': np.zeros((pop_size)),
        },
        'depth': {
            'isolated': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            'ensemble': [None for _ in range(pop_size)],
            },
        'num_nodes':{
            'isolated': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            'ensemble': [None for _ in range(pop_size)],
            },
        'fitness': {
            'isolated': {
                'train': np.full((pop_size, num_pop), np.inf),
                'validation': np.full((pop_size, num_pop), np.inf),
                'test': np.full((pop_size, num_pop), np.inf),
            },
            'ensemble': {
                'train': np.full(pop_size, np.inf),
                'validation': np.full(pop_size, np.inf),
                'test': np.full(pop_size, np.inf),
            },
        },
        'rank': {
            'fitness':{
                'isolated': {
                    'train': [np.full((pop_size), np.inf) for _ in range(num_pop)],
                    'validation': [np.full((pop_size), np.inf) for _ in range(num_pop)],
                    'test': [np.full((pop_size), np.inf) for _ in range(num_pop)],
                    },
                'ensemble': {
                    'train': np.full((pop_size), np.inf),
                    'validation': np.full((pop_size), np.inf),
                    'test': np.full((pop_size), np.inf),
                    },
            },
            'complexity':{
                'isolated': np.full((pop_size, num_pop), np.inf),
                'ensemble': np.full((pop_size), np.inf),
            },
        },
        'all_ensemble':{
            'idx': None,
            'fitness': {
                'train': None,
                'validation': None,
                'test': None,
            },
        },
    }
    
    
    
    

    






