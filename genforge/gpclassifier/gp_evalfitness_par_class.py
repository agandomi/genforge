# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from .gp_evaluate_ensemble_ord_class import gp_evaluate_ensemble_ord
from .gp_getdepth import gp_getdepth
from .gp_evaluate_tree_class import gp_evaluate_tree
from .gp_getcomplexity import gp_getcomplexity
from .gp_getnumnodes import gp_getnumnodes
import copy
import traceback
from .gp_evaluate_softmax_custom import gp_evaluate_softmax_custom

def evaluate_individual(args):
    id_pop, id_ind_range, xtr, xval, xts, ytr, yval, yts, ybin_tr, ybin_val, ybin_ts,\
        function_map_pop, popgp, complexity_measure,\
        num_class, optimizer_type_pop, optimizer_param_pop, initializer_pop,\
        regularization_pop, regularization_rate_pop, batch_size_pop, epochs_pop, random_seed_pop,\
        buffer_size_pop, shuffle_pop, verbose_pop, patience_pop = args
    
    function_map = function_map_pop[id_pop]
    optimizer_type = optimizer_type_pop[id_pop]
    optimizer_param = optimizer_param_pop[id_pop]
    initializer = initializer_pop[id_pop]
    regularization = regularization_pop[id_pop]
    regularization_rate = regularization_rate_pop[id_pop]
    batch_size = batch_size_pop[id_pop]
    epochs = epochs_pop[id_pop]
    random_seed = random_seed_pop
    buffer_size = buffer_size_pop[id_pop]
    shuffle = shuffle_pop[id_pop]
    verbose = verbose_pop[id_pop]
    patience = patience_pop[id_pop]
    
    results_batch = []
    for id_ind in id_ind_range:
        ind = popgp[id_pop][id_ind]
        # Perform the fitness evaluation
        gene_out_tr = np.zeros((xtr.shape[0], len(ind)))
        gene_out_val = np.zeros((xval.shape[0], len(ind))) if yval is not None else None
        gene_out_ts = np.zeros((xts.shape[0], len(ind))) if yts is not None else None
        num_nodes = np.zeros((len(ind)))
        depth = np.zeros((len(ind)))
        complexities_isolated = 0
    
        for id_gene in range(len(ind)):
            # Evaluating genes
            gene_out_tr[:, id_gene] = np.tanh(gp_evaluate_tree(ind[id_gene], xtr, function_map))
            if yval is not None:
                gene_out_val[:, id_gene] = np.tanh(gp_evaluate_tree(ind[id_gene], xval, function_map))
            if yts is not None:
                gene_out_ts[:, id_gene] = np.tanh(gp_evaluate_tree(ind[id_gene], xts, function_map))
            depth[id_gene] = gp_getdepth(ind[id_gene])
            num_nodes[id_gene] = gp_getnumnodes(ind[id_gene])
            if complexity_measure == 1:
                complexities_isolated += gp_getcomplexity(ind[id_gene])
            else:
                complexities_isolated += gp_getnumnodes(ind[id_gene])
    
        args_softmax = ytr, yval, yts, ybin_tr, ybin_val, ybin_ts, num_class, \
            optimizer_type, optimizer_param, initializer,\
            regularization, regularization_rate, batch_size, epochs,\
            patience, random_seed, buffer_size, shuffle, verbose, id_pop, id_ind,\
            gene_out_tr, gene_out_val, gene_out_ts
    
        # Training the softmax
        results = gp_evaluate_softmax_custom(args_softmax)    
        # Assigning the results
        results_batch.append((id_pop, id_ind, gene_out_tr, gene_out_val, gene_out_ts, depth, num_nodes, complexities_isolated, *results))
    
    return results_batch

def gp_evalfitness_par(gp):
    """Evaluate the fitness of individuals (parallel version)."""
    gen = gp.state['generation']
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    function_map = gp.config['nodes']['functions']['function']
    popgp = copy.deepcopy(gp.population)
    optimizer_type = gp.config['softmax']['optimizer_type']
    optimizer_param = gp.config['softmax']['optimizer_param']
    initializer = gp.config['softmax']['initializer']
    regularization = gp.config['softmax']['regularization']
    regularization_rate = gp.config['softmax']['regularization_rate']
    batch_size = gp.config['softmax']['batch_size']
    epochs = gp.config['softmax']['epochs']
    random_seed = gp.config['runcontrol']['random_state']
    buffer_size = gp.config['softmax']['buffer_size']
    shuffle = gp.config['softmax']['shuffle']
    verbose = gp.config['softmax']['verbose']
    patience = gp.config['softmax']['patience']
    complexity_measure = gp.config['fitness']['complexityMeasure']
    num_class = gp.config['runcontrol']['num_class']
    xtr = gp.userdata['xtrain']
    xval = gp.userdata['xval']
    xts = gp.userdata['xtest']
    ytr = gp.userdata['ytrain']
    yval = gp.userdata['yval']
    yts = gp.userdata['ytest']
    ybin_tr = gp.userdata['ybinarytrain']
    ybin_val = gp.userdata['ybinaryval']
    ybin_ts = gp.userdata['ybinarytest']
    
    id_ind_list = []
    for id_pop in range(num_pop):
        id_ind_list.append([])
        for id_ind in range(pop_size):
            if gp.config['runcontrol']['usecache'] and gp.cache['gene_output']['train'][id_pop][id_ind] is not None:
                gp.individuals['gene_output']['train'][id_pop][id_ind] =                copy.deepcopy(gp.cache['gene_output']['train'][id_pop][id_ind])
                gp.individuals['gene_output']['validation'][id_pop][id_ind] =           copy.deepcopy(gp.cache['gene_output']['validation'][id_pop][id_ind])
                gp.individuals['gene_output']['test'][id_pop][id_ind] =                 copy.deepcopy(gp.cache['gene_output']['test'][id_pop][id_ind])
                gp.individuals['prob']['isolated']['train'][id_pop][id_ind] =           copy.deepcopy(gp.cache['prob']['isolated']['train'][id_pop][id_ind])
                gp.individuals['prob']['isolated']['validation'][id_pop][id_ind] =      copy.deepcopy(gp.cache['prob']['isolated']['validation'][id_pop][id_ind])
                gp.individuals['prob']['isolated']['test'][id_pop][id_ind] =            copy.deepcopy(gp.cache['prob']['isolated']['test'][id_pop][id_ind])
                gp.individuals['loss']['isolated']['train'][id_ind, id_pop] =           copy.deepcopy(gp.cache['loss']['isolated']['train'][id_pop][id_ind])
                gp.individuals['loss']['isolated']['validation'][id_ind, id_pop] =      copy.deepcopy(gp.cache['loss']['isolated']['validation'][id_pop][id_ind])
                gp.individuals['loss']['isolated']['test'][id_ind, id_pop] =            copy.deepcopy(gp.cache['loss']['isolated']['test'][id_pop][id_ind])
                gp.individuals['yp']['isolated']['train'][id_pop][id_ind] =             copy.deepcopy(gp.cache['yp']['isolated']['train'][id_pop][id_ind])
                gp.individuals['yp']['isolated']['validation'][id_pop][id_ind] =        copy.deepcopy(gp.cache['yp']['isolated']['validation'][id_pop][id_ind])
                gp.individuals['yp']['isolated']['test'][id_pop][id_ind] =              copy.deepcopy(gp.cache['yp']['isolated']['test'][id_pop][id_ind])
                gp.individuals['fitness']['isolated']['train'][id_ind, id_pop] =        copy.deepcopy(gp.cache['fitness']['isolated']['train'][id_pop][id_ind])
                gp.individuals['fitness']['isolated']['validation'][id_ind, id_pop] =   copy.deepcopy(gp.cache['fitness']['isolated']['validation'][id_pop][id_ind])
                gp.individuals['fitness']['isolated']['test'][id_ind, id_pop] =         copy.deepcopy(gp.cache['fitness']['isolated']['test'][id_pop][id_ind])
                gp.individuals['depth']['isolated'][id_pop][id_ind] =                   copy.deepcopy(gp.cache['depth']['isolated'][id_pop][id_ind])
                gp.individuals['num_nodes']['isolated'][id_pop][id_ind] =               copy.deepcopy(gp.cache['num_nodes']['isolated'][id_pop][id_ind])
                gp.individuals['weight_genes'][id_pop][id_ind] =                        copy.deepcopy(gp.cache['weight_genes'][id_pop][id_ind])
                gp.individuals['complexity']['isolated'][id_ind, id_pop] =              copy.deepcopy(gp.cache['complexity']['isolated'][id_pop][id_ind])
            else:
                id_ind_list[id_pop].append(id_ind)
                
    # Create a ProcessPoolExecutor for multi-core processing
    with ProcessPoolExecutor(max_workers=gp.config['runcontrol']['parallel']['n_jobs']) as executor:
        futures = []
        for id_pop in range(num_pop):
            batch_size_parallel = gp.config['runcontrol']['parallel']['batch_job']  # Processing multiple individuals in a batch to reduce overhead
            for i in range(0, len(id_ind_list[id_pop]), batch_size_parallel):
                id_ind_list_np = np.array(id_ind_list[id_pop])
                id_ind_range = list(id_ind_list_np[list(range(i, min(i + batch_size_parallel, len(id_ind_list_np))))])  # Batch individuals
                args = (id_pop, id_ind_range, xtr, xval, xts, ytr, yval, yts, \
                ybin_tr, ybin_val, ybin_ts, function_map, popgp, complexity_measure,\
                num_class, optimizer_type, optimizer_param, initializer, \
                regularization, regularization_rate, batch_size, epochs, random_seed,\
                buffer_size, shuffle, verbose, patience)
                
                futures.append(executor.submit(evaluate_individual, args))
        
        # Collect results dynamically as they complete
        for future in as_completed(futures):
            try:
                results_batch = future.result()  # Block until each future is complete
                # Handle the results
                for result in results_batch:
                    id_pop, id_ind, gene_out_tr, gene_out_val, gene_out_ts, depth, num_nodes, complexities_isolated, *fitness_results = result
                    # Assign the results back to the gp object
                    gp.individuals['gene_output']['train'][id_pop][id_ind] =                copy.deepcopy(gene_out_tr)
                    gp.individuals['gene_output']['validation'][id_pop][id_ind] =           copy.deepcopy(gene_out_val)
                    gp.individuals['gene_output']['test'][id_pop][id_ind] =                 copy.deepcopy(gene_out_ts)
                    gp.individuals['depth']['isolated'][id_pop][id_ind] =                   copy.deepcopy(depth)
                    gp.individuals['num_nodes']['isolated'][id_pop][id_ind] =               copy.deepcopy(num_nodes)
                    gp.individuals['complexity']['isolated'][id_ind, id_pop] =              copy.deepcopy(complexities_isolated)

                    # Assign fitness results (prob_tr, prob_val, prob_ts, loss_tr, loss_val, loss_ts, yp_tr, yp_val, yp_ts, weight_genes)
                    gp.individuals['prob']['isolated']['train'][id_pop][id_ind] =           copy.deepcopy(fitness_results[0])
                    gp.individuals['prob']['isolated']['validation'][id_pop][id_ind] =      copy.deepcopy(fitness_results[1])
                    gp.individuals['prob']['isolated']['test'][id_pop][id_ind] =            copy.deepcopy(fitness_results[2])
                    gp.individuals['loss']['isolated']['train'][id_ind, id_pop] =           copy.deepcopy(fitness_results[3])
                    gp.individuals['loss']['isolated']['validation'][id_ind, id_pop] =      copy.deepcopy(fitness_results[4])
                    gp.individuals['loss']['isolated']['test'][id_ind, id_pop] =            copy.deepcopy(fitness_results[5])
                    gp.individuals['fitness']['isolated']['train'][id_ind, id_pop] =        copy.deepcopy(fitness_results[3])
                    gp.individuals['fitness']['isolated']['validation'][id_ind, id_pop] =   copy.deepcopy(fitness_results[4])
                    gp.individuals['fitness']['isolated']['test'][id_ind, id_pop] =         copy.deepcopy(fitness_results[5])
                    gp.individuals['yp']['isolated']['train'][id_pop][id_ind] =             copy.deepcopy(fitness_results[6])
                    gp.individuals['yp']['isolated']['validation'][id_pop][id_ind] =        copy.deepcopy(fitness_results[7])
                    gp.individuals['yp']['isolated']['test'][id_pop][id_ind] =              copy.deepcopy(fitness_results[8])
                    gp.individuals['weight_genes'][id_pop][id_ind] =                        copy.deepcopy(fitness_results[9])
            except Exception as exc:
                print(f"Generated an exception: {exc}")
                traceback.print_exc()  # This will print the full stack trace

    # Evaluating the Ensembles
    results_en = gp_evaluate_ensemble_ord(gp)

    # Assigning the results
    en_weight =         copy.deepcopy(results_en[0])
    en_idx =            copy.deepcopy(results_en[1])
    complexity_en =     copy.deepcopy(results_en[2])
    prob_en_tr =        copy.deepcopy(results_en[3])
    prob_en_val =       copy.deepcopy(results_en[4])
    prob_en_ts =        copy.deepcopy(results_en[5])
    loss_en_tr =        copy.deepcopy(results_en[6])
    loss_en_val =       copy.deepcopy(results_en[7])
    loss_en_ts =        copy.deepcopy(results_en[8])
    fit_en_tr =         copy.deepcopy(results_en[9])
    fit_en_val =        copy.deepcopy(results_en[10])
    fit_en_ts =         copy.deepcopy(results_en[11])
    yp_en_tr =          copy.deepcopy(results_en[12])
    yp_en_val =         copy.deepcopy(results_en[13])
    yp_en_ts =          copy.deepcopy(results_en[14])
    depth_en =          copy.deepcopy(results_en[15])
    num_nodes_en =      copy.deepcopy(results_en[16])
    id_ens =            copy.deepcopy(results_en[17])
    fit_ens_tr =        copy.deepcopy(results_en[18])
    fit_ens_val =       copy.deepcopy(results_en[19])
    fit_ens_ts =        copy.deepcopy(results_en[20])
    
    # Assigning the values
    gp.individuals['ensemble_weight'] =                         copy.deepcopy(en_weight)
    gp.individuals['ensemble_idx'] =                            copy.deepcopy(en_idx)
    gp.individuals['complexity']['ensemble'] =                  copy.deepcopy(complexity_en)
    gp.individuals['prob']['ensemble']['train'] =               copy.deepcopy(prob_en_tr)
    gp.individuals['prob']['ensemble']['validation'] =          copy.deepcopy(prob_en_val)
    gp.individuals['prob']['ensemble']['test'] =                copy.deepcopy(prob_en_ts)
    gp.individuals['fitness']['ensemble']['train'] =            copy.deepcopy(fit_en_tr)
    gp.individuals['fitness']['ensemble']['validation'] =       copy.deepcopy(fit_en_val)
    gp.individuals['fitness']['ensemble']['test'] =             copy.deepcopy(fit_en_ts)
    gp.individuals['loss']['ensemble']['train'] =               copy.deepcopy(loss_en_tr)
    gp.individuals['loss']['ensemble']['validation'] =          copy.deepcopy(loss_en_val)
    gp.individuals['loss']['ensemble']['test'] =                copy.deepcopy(loss_en_ts)
    gp.individuals['yp']['ensemble']['train'] =                 copy.deepcopy(yp_en_tr)
    gp.individuals['yp']['ensemble']['validation'] =            copy.deepcopy(yp_en_val)
    gp.individuals['yp']['ensemble']['test'] =                  copy.deepcopy(yp_en_ts)
    gp.individuals['num_nodes']['ensemble'] =                   copy.deepcopy(num_nodes_en)
    gp.individuals['depth']['ensemble'] =                       copy.deepcopy(depth_en)
    gp.individuals['all_ensemble']['idx'] =                     copy.deepcopy(id_ens)
    gp.individuals['all_ensemble']['fitness']['train'] =        copy.deepcopy(fit_ens_tr)
    gp.individuals['all_ensemble']['fitness']['validation'] =   copy.deepcopy(fit_ens_val)
    gp.individuals['all_ensemble']['fitness']['test'] =         copy.deepcopy(fit_ens_ts) 
    
    for id_pop in range(num_pop):
        gp.individuals['rank']['complexity']['isolated'][:, id_pop] = np.argsort(gp.individuals['complexity']['isolated'][:, id_pop]) 
    
    gp.individuals['rank']['complexity']['ensemble'] = np.argsort(gp.individuals['complexity']['ensemble'])
    
    if gp.config['runcontrol']['minimisation']:
        gp.individuals['rank']['fitness']['ensemble']['train'] = np.argsort(gp.individuals['fitness']['ensemble']['train'])
        if xval is not None:
            gp.individuals['rank']['fitness']['ensemble']['validation'] = np.argsort(gp.individuals['fitness']['ensemble']['validation'])
        if xts is not None:
            gp.individuals['rank']['fitness']['ensemble']['test'] = np.argsort(gp.individuals['fitness']['ensemble']['test'])
        
        for id_pop in range(num_pop):
            gp.individuals['rank']['fitness']['isolated']['train'][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['train'][:, id_pop])
            if xval is not None:
                gp.individuals['rank']['fitness']['isolated']['validation'][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['validation'][:, id_pop])
            if xts is not None:
                gp.individuals['rank']['fitness']['isolated']['test'][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['test'][:, id_pop])
    else:
        gp.individuals['rank']['fitness']['ensemble']['train'] = np.argsort(-gp.individuals['fitness']['ensemble']['train'])
        if xval is not None:
            gp.individuals['rank']['fitness']['ensemble']['validation'] = np.argsort(-gp.individuals['fitness']['ensemble']['validation'])
        if xts is not None:
            gp.individuals['rank']['fitness']['ensemble']['test'] = np.argsort(-gp.individuals['fitness']['ensemble']['test'])
        
        for id_pop in range(num_pop):
            gp.individuals['rank']['fitness']['isolated']['train'][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['train'][:, id_pop])
            if xval is not None:
                gp.individuals['rank']['fitness']['isolated']['validation'][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['validation'][:, id_pop])
            if xts is not None:
                gp.individuals['rank']['fitness']['isolated']['test'][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['test'][:, id_pop])


    # Assigning Fitness value
    gp.fitness['values'] = copy.deepcopy(gp.individuals['fitness']['ensemble']['train'])
    gp.fitness['complexities'] = copy.deepcopy(gp.individuals['complexity']['ensemble'])
    # Assign the tracking parameters
    if gp.config['runcontrol']['track_individuals']:
        gp.track['complexity']['isolated'][gen] =                   copy.deepcopy(gp.individuals['complexity']['isolated'])
        gp.track['complexity']['ensemble'][gen] =                   copy.deepcopy(gp.individuals['complexity']['ensemble'])
        gp.track['fitness']['isolated']['train'][gen] =             copy.deepcopy(gp.individuals['fitness']['isolated']['train'])
        gp.track['fitness']['isolated']['validation'][gen] =        copy.deepcopy(gp.individuals['fitness']['isolated']['validation'])
        gp.track['fitness']['isolated']['test'][gen] =              copy.deepcopy(gp.individuals['fitness']['isolated']['test'])
        gp.track['fitness']['ensemble']['train'][gen] =             copy.deepcopy(gp.individuals['fitness']['ensemble']['train'])
        gp.track['fitness']['ensemble']['validation'][gen] =        copy.deepcopy(gp.individuals['fitness']['ensemble']['validation'])
        gp.track['fitness']['ensemble']['test'][gen] =              copy.deepcopy(gp.individuals['fitness']['ensemble']['test'])
        gp.track['std_fitness']['isolated']['train'][gen] =         np.std(gp.individuals['fitness']['isolated']['train'], axis = 0)
        gp.track['std_fitness']['isolated']['validation'][gen] =    np.std(gp.individuals['fitness']['isolated']['validation'], axis = 0) 
        gp.track['std_fitness']['isolated']['test'][gen] =          np.std(gp.individuals['fitness']['isolated']['test'], axis = 0) 
        gp.track['std_fitness']['ensemble']['train'][gen] =         np.std(gp.individuals['fitness']['ensemble']['train'])
        gp.track['std_fitness']['ensemble']['validation'][gen] =    np.std(gp.individuals['fitness']['ensemble']['validation'])
        gp.track['std_fitness']['ensemble']['test'][gen] =          np.std(gp.individuals['fitness']['ensemble']['test'])
        gp.track['mean_fitness']['isolated']['train'][gen] =        np.mean(gp.individuals['fitness']['isolated']['train'], axis = 0)  
        gp.track['mean_fitness']['isolated']['validation'][gen] =   np.mean(gp.individuals['fitness']['isolated']['validation'], axis = 0) 
        gp.track['mean_fitness']['isolated']['test'][gen] =         np.mean(gp.individuals['fitness']['isolated']['test'], axis = 0) 
        gp.track['mean_fitness']['ensemble']['train'][gen] =        np.mean(gp.individuals['fitness']['ensemble']['train'])
        gp.track['mean_fitness']['ensemble']['validation'][gen] =   np.mean(gp.individuals['fitness']['ensemble']['validation'])
        gp.track['mean_fitness']['ensemble']['test'][gen] =         np.mean(gp.individuals['fitness']['ensemble']['test'])
        gp.track['ensemble_idx'][gen] =                             copy.deepcopy(gp.individuals['ensemble_idx'])
        gp.track['depth']['isolated'][gen] =                        copy.deepcopy(gp.individuals['depth']['isolated'])
        gp.track['depth']['ensemble'][gen] =                        copy.deepcopy(gp.individuals['depth']['ensemble'])
        gp.track['num_nodes']['isolated'][gen] =                    copy.deepcopy(gp.individuals['num_nodes']['isolated'])
        gp.track['num_nodes']['ensemble'][gen] =                    copy.deepcopy(gp.individuals['num_nodes']['ensemble'])
        gp.track['all_ensemble']['idx'][gen] =                      copy.deepcopy(id_ens)
        gp.track['all_ensemble']['fitness']['train'][gen] =         copy.deepcopy(fit_ens_tr)
        gp.track['all_ensemble']['fitness']['validation'][gen] =    copy.deepcopy(fit_ens_val)
        gp.track['all_ensemble']['fitness']['test'][gen] =          copy.deepcopy(fit_ens_ts)               
        gp.track['yp']['isolated']['train'][gen] =                  copy.deepcopy(gp.individuals['yp']['isolated']['train'])
        gp.track['yp']['isolated']['validation'][gen] =             copy.deepcopy(gp.individuals['yp']['isolated']['validation'])
        gp.track['yp']['isolated']['test'][gen] =                   copy.deepcopy(gp.individuals['yp']['isolated']['test'])
        gp.track['yp']['ensemble']['train'][gen] =                  copy.deepcopy(gp.individuals['yp']['ensemble']['train'])
        gp.track['yp']['ensemble']['validation'][gen] =             copy.deepcopy(gp.individuals['yp']['ensemble']['validation'])
        gp.track['yp']['ensemble']['test'][gen] =                   copy.deepcopy(gp.individuals['yp']['ensemble']['test'])
        gp.track['prob']['isolated']['train'][gen] =                  copy.deepcopy(gp.individuals['prob']['isolated']['train'])
        gp.track['prob']['isolated']['validation'][gen] =             copy.deepcopy(gp.individuals['prob']['isolated']['validation'])
        gp.track['prob']['isolated']['test'][gen] =                   copy.deepcopy(gp.individuals['prob']['isolated']['test'])
        gp.track['prob']['ensemble']['train'][gen] =                  copy.deepcopy(gp.individuals['prob']['ensemble']['train'])
        gp.track['prob']['ensemble']['validation'][gen] =             copy.deepcopy(gp.individuals['prob']['ensemble']['validation'])
        gp.track['prob']['ensemble']['test'][gen] =                   copy.deepcopy(gp.individuals['prob']['ensemble']['test'])
        
        for id_pop in range(num_pop):
            gp.track['rank']['complexity']['isolated'][gen][:, id_pop] = np.argsort(gp.individuals['complexity']['isolated'][:, id_pop]) 
        
        gp.track['rank']['complexity']['ensemble'][gen] = np.argsort(gp.individuals['complexity']['ensemble'])
        
        if gp.config['runcontrol']['minimisation']:
            gp.track['rank']['fitness']['ensemble']['train'][gen] = np.argsort(gp.individuals['fitness']['ensemble']['train'])
            if xval is not None:
                gp.track['rank']['fitness']['ensemble']['validation'][gen] = np.argsort(gp.individuals['fitness']['ensemble']['validation'])
            if xts is not None:
                gp.track['rank']['fitness']['ensemble']['test'][gen] = np.argsort(gp.individuals['fitness']['ensemble']['test'])
            
            for id_pop in range(num_pop):
                gp.track['rank']['fitness']['isolated']['train'][gen][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['train'][:, id_pop])
                if xval is not None:
                    gp.track['rank']['fitness']['isolated']['validation'][gen][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['validation'][:, id_pop])
                if xts is not None:
                    gp.track['rank']['fitness']['isolated']['test'][gen][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['test'][:, id_pop])
        else:
            gp.track['rank']['fitness']['ensemble']['train'][gen] = np.argsort(-gp.individuals['fitness']['ensemble']['train'])
            if xval is not None:
                gp.track['rank']['fitness']['ensemble']['validation'][gen] = np.argsort(-gp.individuals['fitness']['ensemble']['validation'])
            if xts is not None:
                gp.track['rank']['fitness']['ensemble']['test'][gen] = np.argsort(-gp.individuals['fitness']['ensemble']['test'])
            
            for id_pop in range(num_pop):
                gp.track['rank']['fitness']['isolated']['train'][gen][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['train'][:, id_pop])
                if xval is not None:
                    gp.track['rank']['fitness']['isolated']['validation'][gen][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['validation'][:, id_pop])
                if xts is not None:
                    gp.track['rank']['fitness']['isolated']['test'][gen][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['test'][:, id_pop])
    
