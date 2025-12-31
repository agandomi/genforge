# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi


def gp_cache(gp):
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    
    if gp.config['runcontrol']['usecache']:
        gp.cache = {
            'weight_genes': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            'gene_output': {
                'train': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                'validation': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                'test': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            },
            'gene_penalty': {
                'train': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                'validation': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                'test': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            },
            'loss': {
                'isolated': {
                    'train': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                    'validation': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                    'test': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                },
            },
            'objective': {
                'isolated': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            },
            'yp': {
                'isolated': {
                    'train': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                    'validation': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                    'test': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                },
            },
            'complexity': {
                'isolated': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            },
            'depth': {
                'isolated': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            },
            'num_nodes':{
                'isolated': [[None for _ in range(pop_size)] for _ in range(num_pop)],
            },
            'fitness': {
                'isolated': {
                    'train': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                    'validation': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                    'test': [[None for _ in range(pop_size)] for _ in range(num_pop)],
                },
            },
        }