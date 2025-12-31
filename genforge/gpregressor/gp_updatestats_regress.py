# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
import copy
from .gp_copydetect import gp_copydetect
import time

def gp_updatestats(gp):
    num_generations = gp.config['runcontrol']['num_generations']
    num_pop = gp.config['runcontrol']['num_pop']
    gen = gp.state['generation']
    tolfit = gp.config['runcontrol']['tolfit'] 
    stallgen = gp.config['runcontrol']['stallgen']
    adaptgen = gp.config['runcontrol']['adaptgen']
    # fit_iso = copy.deepcopy(gp.state['best']['fitness']['isolated']['train'])
    fit_en = copy.deepcopy(gp.state['best']['objective']['ensemble'])
    individual_best_iso = copy.deepcopy( gp.state['best']['individual']['isolated'] )
    individual_best_en = copy.deepcopy( gp.state['best']['individual']['ensemble'] )
    # com_iso = gp.state['best']['complexity']['isolated'].deepcopy()
    # com_en = gp.state['best']['complexity']['ensemble'].deepcopy()
    xval = copy.deepcopy(gp.userdata['xval'])
    xtest = copy.deepcopy(gp.userdata['xtest'])
    
    
    best_id_fit_iso = list()
    if gp.config['runcontrol']['minimisation']:
        for id_pop in range(num_pop):
            gp.state['best']['objective']['isolated'][id_pop] = \
                np.concatenate(( gp.state['best']['objective']['isolated'][id_pop],\
                [np.nanmin(gp.individuals['objective']['isolated'][:, id_pop])] ))   
                
            best_id_fit_iso.append( list(np.where(gp.individuals['objective']['isolated'][:, id_pop] ==
                                    np.nanmin(gp.individuals['objective']['isolated'][:, id_pop]) )[0]) )   
        gp.state['best']['objective']['ensemble'] = \
            np.concatenate(( gp.state['best']['objective']['ensemble'],\
            [np.nanmin(gp.individuals['objective']['ensemble'])] ))   
                
        best_id_fit_en = list(np.where(gp.individuals['objective']['ensemble'] ==
                                    np.nanmin(gp.individuals['objective']['ensemble']) )[0])
    else:
        for id_pop in range(num_pop):
            gp.state['best']['objective']['isolated'][id_pop] = \
                np.concatenate(( gp.state['best']['objective']['isolated'][id_pop],\
                [np.nanmax(gp.individuals['objective']['isolated'][:, id_pop])] ))   
                
            best_id_fit_iso.append( list(np.where(gp.individuals['objective']['isolated'][:, id_pop] ==
                                    np.nanmax(gp.individuals['objective']['isolated'][:, id_pop]) )[0]) )   
        gp.state['best']['objective']['ensemble'] = \
            np.concatenate(( gp.state['best']['objective']['ensemble'],\
            [np.nanmax(gp.individuals['objective']['ensemble'])] ))    
        
        best_id_fit_en = list(np.where(gp.individuals['objective']['ensemble'] ==
                                    np.nanmax(gp.individuals['objective']['ensemble']) )[0])
    
    best_idx_fit_iso = list()
    for id_pop in range(num_pop):
        best_idx_fit_iso.append(best_id_fit_iso[id_pop][int(
            np.where(gp.individuals['complexity']['isolated'][best_id_fit_iso[id_pop], id_pop] ==
                    np.nanmin( gp.individuals['complexity']['isolated'][best_id_fit_iso[id_pop], id_pop] ))[0][0])])
        
        gp.state['best']['complexity']['isolated'][id_pop] = \
            np.concatenate(( gp.state['best']['complexity']['isolated'][id_pop],\
            [gp.individuals['complexity']['isolated'][best_idx_fit_iso[id_pop], id_pop]] ))
                
        gp.state['best']['individual']['isolated'][id_pop].append( copy.deepcopy(gp.population[id_pop][best_idx_fit_iso[id_pop]]) )
        
        gp.state['best']['fitness']['isolated']['train'][id_pop] = \
            np.concatenate(( gp.state['best']['fitness']['isolated']['train'][id_pop],\
            [gp.individuals['fitness']['isolated']['train'][best_idx_fit_iso[id_pop], id_pop]] ))
        if xval is not None:
            gp.state['best']['fitness']['isolated']['validation'][id_pop] = \
                np.concatenate(( gp.state['best']['fitness']['isolated']['validation'][id_pop],\
                [gp.individuals['fitness']['isolated']['validation'][best_idx_fit_iso[id_pop], id_pop]] ))
        if xtest is not None:
            gp.state['best']['fitness']['isolated']['test'][id_pop] = \
                np.concatenate(( gp.state['best']['fitness']['isolated']['test'][id_pop],\
                [gp.individuals['fitness']['isolated']['test'][best_idx_fit_iso[id_pop], id_pop]] ))
                    
        gp.state['best']['penalty']['isolated']['train'][id_pop] = \
            np.append( gp.state['best']['penalty']['isolated']['train'][id_pop],\
            gp.individuals['penalty']['isolated']['train'][best_idx_fit_iso[id_pop], id_pop] )
        if xval is not None:
            gp.state['best']['penalty']['isolated']['validation'][id_pop] = \
                np.append( gp.state['best']['penalty']['isolated']['validation'][id_pop],\
                gp.individuals['penalty']['isolated']['validation'][best_idx_fit_iso[id_pop], id_pop] )
        if xtest is not None:
            gp.state['best']['penalty']['isolated']['test'][id_pop] = \
                np.append( gp.state['best']['penalty']['isolated']['test'][id_pop],\
                gp.individuals['penalty']['isolated']['test'][best_idx_fit_iso[id_pop], id_pop] )
        
    best_idx_fit_en = best_id_fit_en[int(np.where(gp.individuals['complexity']['ensemble'][best_id_fit_en] ==
                np.nanmin( gp.individuals['complexity']['ensemble'][best_id_fit_en] ))[0][0])]
    
    gp.state['best']['complexity']['ensemble'] = \
        np.concatenate(( gp.state['best']['complexity']['ensemble'],\
        [gp.individuals['complexity']['ensemble'][best_idx_fit_en]] ))
    
    gp.state['best']['fitness']['ensemble']['train'] = \
        np.concatenate(( gp.state['best']['fitness']['ensemble']['train'],\
        [gp.individuals['fitness']['ensemble']['train'][best_idx_fit_en]] ))
    if xval is not None:
        gp.state['best']['fitness']['ensemble']['validation'] = \
            np.concatenate(( gp.state['best']['fitness']['ensemble']['validation'],\
            [gp.individuals['fitness']['ensemble']['validation'][best_idx_fit_en]] ))
                
    if xtest is not None:
        gp.state['best']['fitness']['ensemble']['test'] = \
            np.concatenate(( gp.state['best']['fitness']['ensemble']['test'],\
            [gp.individuals['fitness']['ensemble']['test'][best_idx_fit_en]] ))
                
    gp.state['best']['penalty']['ensemble']['train'] = \
        np.concatenate(( gp.state['best']['penalty']['ensemble']['train'],\
        [gp.individuals['penalty']['ensemble']['train'][best_idx_fit_en]] ))
    if xval is not None:
        gp.state['best']['penalty']['ensemble']['validation'] = \
            np.concatenate(( gp.state['best']['penalty']['ensemble']['validation'],\
            [gp.individuals['penalty']['ensemble']['validation'][best_idx_fit_en]] ))
                
    if xtest is not None:
        gp.state['best']['penalty']['ensemble']['test'] = \
            np.concatenate(( gp.state['best']['penalty']['ensemble']['test'],\
            [gp.individuals['penalty']['ensemble']['test'][best_idx_fit_en]] ))
    
    for id_pop in range(num_pop):
        gp.state['best']['idx']['isolated'][id_pop] = np.concatenate((gp.state['best']['idx']['isolated'][id_pop],\
                [best_idx_fit_iso[id_pop]]))
        
    gp.state['best']['idx']['ensemble'].append(list(copy.deepcopy(gp.individuals['ensemble_idx'][best_idx_fit_en, :])))
    gp.state['best']['ensemble_weight'].append(list(copy.deepcopy(gp.individuals['ensemble_weight'][best_idx_fit_en])))
    
    weight_genes_iso = list()
    weight_genes_en = list()
    individual_iso = list()
    individual_en = list()
    num_node_iso = list()
    num_node_en = list()
    depth_iso = list()
    depth_en = list()
    for id_pop in range(num_pop):
        weight_genes_iso.append(copy.deepcopy(gp.individuals['weight_genes'][id_pop][best_idx_fit_iso[id_pop]]))
        weight_genes_en.append(copy.deepcopy(gp.individuals['weight_genes'][id_pop][int(gp.individuals['ensemble_idx'][best_idx_fit_en, id_pop])]))
        individual_iso.append(copy.deepcopy(gp.population[id_pop][best_idx_fit_iso[id_pop]]))
        individual_en.append(copy.deepcopy(gp.population[id_pop][int(gp.individuals['ensemble_idx'][best_idx_fit_en, id_pop])]))
        num_node_iso.append(copy.deepcopy( gp.individuals['num_nodes']['isolated'][id_pop][best_idx_fit_iso[id_pop]] ))
        num_node_en.append(copy.deepcopy( gp.individuals['num_nodes']['ensemble'][best_idx_fit_en] ))
        depth_iso.append(copy.deepcopy( gp.individuals['depth']['isolated'][id_pop][best_idx_fit_iso[id_pop]] ))
        depth_en.append(copy.deepcopy( gp.individuals['depth']['ensemble'][best_idx_fit_en] ))
    
    gp.state['best']['weight_genes']['isolated'].append(copy.deepcopy(weight_genes_iso))
    gp.state['best']['weight_genes']['ensemble'].append(copy.deepcopy(weight_genes_en))
    gp.state['best']['individual']['isolated'].append(copy.deepcopy(individual_iso))
    gp.state['best']['individual']['ensemble'].append(copy.deepcopy(individual_en))
    gp.state['best']['num_nodes']['isolated'].append(copy.deepcopy(num_node_iso))
    gp.state['best']['num_nodes']['ensemble'].append(copy.deepcopy(num_node_en))
    gp.state['best']['depth']['isolated'].append(copy.deepcopy(depth_iso))
    gp.state['best']['depth']['ensemble'].append(copy.deepcopy(depth_en))
    
    found_at_gen_iso = np.zeros((1,num_pop))
    copy_en = list()
    for id_pop in range(num_pop):
        if gen > 0:
            copy_en.append(gp_copydetect(individual_best_en[-1], individual_en[id_pop], id_pop))
            if gp_copydetect(individual_best_iso[-1], individual_iso[id_pop], id_pop):
                found_at_gen_iso[0,id_pop] = gp.state['best']['found_at_generation']['isolated'][-1, id_pop]
            else:
                found_at_gen_iso[0,id_pop] = gen
        else:
            found_at_gen_iso[0,id_pop] = gen
            copy_en.append(False)
                
    gp.state['best']['found_at_generation']['isolated'] = np.concatenate((gp.state['best']['found_at_generation']['isolated'],\
        found_at_gen_iso), axis = 0)
    
    if not np.all(copy_en) or gen == 0:
        gp.state['best']['found_at_generation']['ensemble'] = np.concatenate((gp.state['best']['found_at_generation']['ensemble'], [gen]))
    elif np.all(copy_en):
        gp.state['best']['found_at_generation']['ensemble'] = np.concatenate((gp.state['best']['found_at_generation']['ensemble'],\
            [gp.state['best']['found_at_generation']['ensemble'][-1]]))
        
    stall_gen = 0
    if gen > 0:
        if gp.config['runcontrol']['minimisation']: 
            if gp.state['best']['objective']['ensemble'][-1] < fit_en[-1] - tolfit:
                stall_gen = 0
                if gp.state['adaptinjected']:
                    gp.state['adaptinjected'] = False
            else:
                stall_gen = gp.state['stallgen'] + 1
        else:
            if gp.state['best']['objective']['ensemble'][-1]  > fit_en[-1]  + tolfit:
                stall_gen = 0
                if gp.state['adaptinjected']:
                    gp.state['adaptinjected'] = False
            else:
                stall_gen = gp.state['stallgen'] + 1
        
    gp.state['stallgen'] = stall_gen   
    
    if gen == num_generations:
        gp.state['terminate'] = True
        gp.state['run_completed'] = True
    elif gp.state['stallgen'] == stallgen:
        gp.state['terminate'] = True
        gp.state['run_completed'] = True
    
    for id_pop in range(num_pop):
        gp.state['std_fitness']['isolated']['train'][id_pop] =\
            np.concatenate(( gp.state['std_fitness']['isolated']['train'][id_pop],
            [np.std(gp.individuals['fitness']['isolated']['train'][:, id_pop])] ))
        gp.state['mean_fitness']['isolated']['train'][id_pop] =\
            np.concatenate(( gp.state['mean_fitness']['isolated']['train'][id_pop],
            [np.mean(gp.individuals['fitness']['isolated']['train'][:, id_pop])] ))
            
        if xval is not None:
            gp.state['std_fitness']['isolated']['validation'][id_pop] =\
                np.concatenate(( gp.state['std_fitness']['isolated']['validation'][id_pop],
                [np.std(gp.individuals['fitness']['isolated']['validation'][:, id_pop])] ))
            gp.state['mean_fitness']['isolated']['validation'][id_pop] =\
                np.concatenate(( gp.state['mean_fitness']['isolated']['validation'][id_pop],
                [np.mean(gp.individuals['fitness']['isolated']['validation'][:, id_pop])] ))
                
        if xtest is not None:
            gp.state['std_fitness']['isolated']['test'][id_pop] =\
                np.concatenate(( gp.state['std_fitness']['isolated']['test'][id_pop],
                [np.std(gp.individuals['fitness']['isolated']['test'][:, id_pop])] ))
            gp.state['mean_fitness']['isolated']['test'][id_pop] =\
                np.concatenate(( gp.state['mean_fitness']['isolated']['test'][id_pop],
                [np.mean(gp.individuals['fitness']['isolated']['test'][:, id_pop])] ))
                
    gp.state['std_fitness']['ensemble']['train'] =\
        np.concatenate(( gp.state['std_fitness']['ensemble']['train'],
        [np.std(gp.individuals['fitness']['ensemble']['train'])] ))
    gp.state['mean_fitness']['ensemble']['train'] =\
        np.concatenate(( gp.state['mean_fitness']['ensemble']['train'],
        [np.mean(gp.individuals['fitness']['ensemble']['train'])] ))
        
    if xval is not None:
        gp.state['std_fitness']['ensemble']['validation'] =\
            np.concatenate(( gp.state['std_fitness']['ensemble']['validation'],
            [np.std(gp.individuals['fitness']['ensemble']['validation'])] ))
        gp.state['mean_fitness']['ensemble']['validation'] =\
            np.concatenate(( gp.state['mean_fitness']['ensemble']['validation'],
            [np.mean(gp.individuals['fitness']['ensemble']['validation'])] ))
            
    if xtest is not None:
        gp.state['std_fitness']['ensemble']['test'] =\
            np.concatenate(( gp.state['std_fitness']['ensemble']['test'],
            [np.std(gp.individuals['fitness']['ensemble']['test'])] ))
        gp.state['mean_fitness']['ensemble']['test'] =\
            np.concatenate(( gp.state['mean_fitness']['ensemble']['test'],
            [np.mean(gp.individuals['fitness']['ensemble']['test'])] ))
            
    if gp.config['runcontrol']['usecache']:
        gp.state['cache'].append(copy.deepcopy(gp.cache))
        
    
    gp.state['time'].append(time.time())
    gp.state['TimeElapsed'].append(gp.state['time'][-1] - gp.state['time'][0])
            