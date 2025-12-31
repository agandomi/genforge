# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
import time

def gp_state_init_ensemble(gp):
    """Initializes a run."""
    num_pop = gp.config['runcontrol']['num_pop']
    # pop_size = gp.config['runcontrol']['pop_size']
    # Initialize some state and tracker variables
    gp.state = {
        'generation': -1,
        'best': {
            'fitness': {
                'isolated': {
                    'train': [np.empty((0)) for _ in range(num_pop)],  # Fixed
                    'validation': [np.empty((0)) for _ in range(num_pop)],  # Fixed
                    'test': [np.empty((0)) for _ in range(num_pop)]  # Fixed
                },
                'ensemble': {'train': np.empty((0)), 'validation': np.empty((0)), 'test': np.empty((0))}
            },
            'complexity': {'isolated': [np.empty((0)) for _ in range(num_pop)], 'ensemble': np.empty((0))},  # Fixed
            'depth': {'isolated': list(), 'ensemble': list()},
            'num_nodes':{'isolated': list(), 'ensemble': list()},
            'idx': {'isolated': [np.empty((0)) for _ in range(num_pop)], 'ensemble': list()}, # Fixed
            'ensemble_weight': list(),
            'weight_genes': {'isolated': list(), 'ensemble': list()},
            'individual': {'isolated': [list() for _ in range(num_pop)], 'ensemble': [list() for _ in range(num_pop)]},
            'found_at_generation': {'isolated': np.empty((0, num_pop)), 'ensemble': np.empty((0))},  # Fixed
        },
        'stallgen': 0,
        'run_completed': False,
        'std_fitness': {
            'isolated': {
                'train': [np.empty((0)) for _ in range(num_pop)],  # Fixed
                'validation': [np.empty((0)) for _ in range(num_pop)],  # Fixed
                'test': [np.empty((0)) for _ in range(num_pop)]  # Fixed
            },
            'ensemble': {'train': np.empty((0)), 'validation': np.empty((0)), 'test': np.empty((0))}
        },
        'mean_fitness': {
            'isolated': {
                'train': [np.empty((0)) for _ in range(num_pop)],  # Fixed
                'validation': [np.empty((0)) for _ in range(num_pop)],  # Fixed
                'test': [np.empty((0)) for _ in range(num_pop)]  # Fixed
            },
            'ensemble': {'train': np.empty((0)), 'validation': np.empty((0)), 'test': np.empty((0))}
        },
        'terminate': False,
        'cache': list(),
        'run': None,
        'time': [time.time()],
        'TimeElapsed': [0],
    }