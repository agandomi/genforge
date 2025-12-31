# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import copy
def gp_plot_prepare_data(gp):
    
    data_keys = ['train']
    if gp.userdata['xval'] is not None:
        data_keys.append('validation')
    if gp.userdata['xtest'] is not None:
        data_keys.append('test')
    
    if 'data' not in gp.plot:
        gp.plot['data'] = {
            'rank': {
                'fitness':{
                    'isolated': {
                        'train': [],
                        'validation': [],
                        'test': [],
                        },
                    'ensemble': {
                        'train': [],
                        'validation': [],
                        'test': [],
                        },
                    },
                'complexity':{
                    'isolated': [],
                    'ensemble': [],
                    },
                },
            'all_ensemble':{
                'idx': [],
                'fitness': {
                    'train': [],
                    'validation': [],
                    'test': [],
                    },
                },
            }
    
    ##################################
    gp.plot['data']['rank']['complexity']['isolated'].append(copy.deepcopy(gp.individuals['rank']['complexity']['isolated']))
    gp.plot['data']['rank']['complexity']['ensemble'].append(copy.deepcopy(gp.individuals['rank']['complexity']['ensemble']))
    gp.plot['data']['all_ensemble']['idx'].append(copy.deepcopy(gp.individuals['all_ensemble']['idx']))
    
    for key in data_keys:
        gp.plot['data']['rank']['fitness']['isolated'][key].append(copy.deepcopy(gp.individuals['rank']['fitness']['isolated'][key]))
        gp.plot['data']['rank']['fitness']['ensemble'][key].append(copy.deepcopy(gp.individuals['rank']['fitness']['ensemble'][key]))
        gp.plot['data']['all_ensemble']['fitness'][key].append(copy.deepcopy(gp.individuals['all_ensemble']['fitness'][key]))
    