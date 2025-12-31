# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
from .gp_evaluate_tree_regress import gp_evaluate_tree
from .gp_getcomplexity import gp_getcomplexity
from .gp_getnumnodes import gp_getnumnodes
from .gp_evaluate_ensemble_ord_regress import gp_evaluate_ensemble_ord
from .gp_getdepth import gp_getdepth
import copy
from .gp_evaluate_linear_regression_crossfold import gp_evaluate_linear_regression_crossfold
#from .utils import tree2evalstr, getcomplexity, getnumnodes

def gp_evalfitness_ord(gp):
    """Evaluate the fitness of individuals (ordinary version)."""
    gen = gp.state['generation']
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    popgp = copy.deepcopy(gp.population)
    complexity_measure = gp.config['fitness']['complexityMeasure']
    function_map = gp.config['nodes']['functions']['function']
    xtr = gp.userdata['xtrain']
    xval = gp.userdata['xval']
    xts = gp.userdata['xtest']
    ytr = gp.userdata['ytrain']
    yval = gp.userdata['yval']
    yts = gp.userdata['ytest']
    ztr = gp.state['ztrain']
    zval = gp.state['zval']
    zts = gp.state['ztest']
    

    for id_pop in range(num_pop):
        pop = popgp[id_pop]
        # elasticnetcv parameters
        l1_ratio = gp.config['linregression']['l1_ratio'][id_pop]
        alphas = gp.config['linregression']['alphas'][id_pop]
        n_alphas = gp.config['linregression']['n_alphas'][id_pop]
        eps = gp.config['linregression']['eps'][id_pop]
        fit_intercept = gp.config['linregression']['fit_intercept'][id_pop]
        copy_x = gp.config['linregression']['copy_x'][id_pop]
        max_iter = gp.config['linregression']['max_iter'][id_pop]
        tol = gp.config['linregression']['tol'][id_pop]
        cv = gp.config['linregression']['cv'][id_pop]
        n_jobs = gp.config['linregression']['n_jobs'][id_pop]
        verbose = gp.config['linregression']['verbose'][id_pop]
        positive = gp.config['linregression']['positive'][id_pop]
        selection = gp.config['linregression']['selection'][id_pop]
        random_seed = gp.config['runcontrol']['random_state']
        # Update state to index of the individual that is about to be evaluated
        res_indiv = []
        objective = []
        for id_ind in range(pop_size):
            if gp.config['runcontrol']['usecache'] and gp.cache['gene_output']['train'][id_pop][id_ind] is not None:
                gene_out_tr =               copy.deepcopy(gp.cache['gene_output']['train'][id_pop][id_ind])
                gene_out_val =              copy.deepcopy(gp.cache['gene_output']['validation'][id_pop][id_ind])
                gene_out_ts =               copy.deepcopy(gp.cache['gene_output']['test'][id_pop][id_ind])
                gene_penalty_tr =           copy.deepcopy(gp.cache['gene_penalty']['train'][id_pop][id_ind])
                gene_penalty_val =          copy.deepcopy(gp.cache['gene_penalty']['validation'][id_pop][id_ind])
                gene_penalty_ts =           copy.deepcopy(gp.cache['gene_penalty']['test'][id_pop][id_ind])
                loss_tr =                   copy.deepcopy(gp.cache['loss']['isolated']['train'][id_pop][id_ind])
                loss_val =                  copy.deepcopy(gp.cache['loss']['isolated']['validation'][id_pop][id_ind])
                loss_ts =                   copy.deepcopy(gp.cache['loss']['isolated']['test'][id_pop][id_ind])
                yp_tr =                     copy.deepcopy(gp.cache['yp']['isolated']['train'][id_pop][id_ind])
                yp_val =                    copy.deepcopy(gp.cache['yp']['isolated']['validation'][id_pop][id_ind])
                yp_ts =                     copy.deepcopy(gp.cache['yp']['isolated']['test'][id_pop][id_ind])
                fit_tr =                    copy.deepcopy(gp.cache['fitness']['isolated']['train'][id_pop][id_ind])
                fit_val =                   copy.deepcopy(gp.cache['fitness']['isolated']['validation'][id_pop][id_ind])
                fit_ts =                    copy.deepcopy(gp.cache['fitness']['isolated']['test'][id_pop][id_ind])
                depth =                     copy.deepcopy(gp.cache['depth']['isolated'][id_pop][id_ind])
                num_nodes =                 copy.deepcopy(gp.cache['num_nodes']['isolated'][id_pop][id_ind])
                weight_genes =              copy.deepcopy(gp.cache['weight_genes'][id_pop][id_ind])
                complexities_isolated =     copy.deepcopy(gp.cache['complexity']['isolated'][id_pop][id_ind])
                obj =                       copy.deepcopy(gp.cache['objective']['isolated'][id_pop][id_ind])
            else:
                ind = pop[id_ind]
                gene_out_tr = np.zeros((xtr.shape[0], len(ind)))
                gene_penalty_tr = np.zeros((xtr.shape[0], len(ind)))
                
                if xval is not None:
                    gene_out_val = np.zeros((xval.shape[0], len(ind)))
                    gene_penalty_val = np.zeros((xval.shape[0], len(ind)))
                else:
                    gene_out_val = None
                    gene_penalty_val = None
                
                if xts is not None:
                    gene_out_ts = np.zeros((xts.shape[0], len(ind)))
                    gene_penalty_ts = np.zeros((xts.shape[0], len(ind)))
                else:
                    gene_out_ts = None
                    gene_penalty_ts = None
                    
                num_nodes = np.zeros((len(ind)))
                depth = np.zeros((len(ind)))
                complexities_isolated = 0
                for id_gene in range(len(ind)):
                    # Evaluating genes
                    gene_out_tr [:,id_gene], gene_penalty_tr[:, id_gene] = gp_evaluate_tree(ind [id_gene], xtr, ztr, function_map[id_pop])
                    if xval is not None:
                        gene_out_val [:,id_gene], gene_penalty_val[:, id_gene] = gp_evaluate_tree(ind [id_gene], xval, zval, function_map[id_pop])
                    else:
                        gene_out_val = None
                        gene_penalty_val = None
                        
                    if xts is not None:
                        gene_out_ts [:,id_gene], gene_penalty_ts[:, id_gene] = gp_evaluate_tree(ind [id_gene], xts, zts, function_map[id_pop])
                    else:
                        gene_out_ts = None
                        gene_penalty_ts = None
                    
                    depth[id_gene] = gp_getdepth(ind[id_gene])
                    num_nodes[id_gene] = gp_getnumnodes(ind[id_gene])
                    if complexity_measure == 1:
                        complexities_isolated += gp_getcomplexity(ind[id_gene])
                    else:
                        complexities_isolated += gp_getnumnodes(ind[id_gene])
                    
                args = ytr, yval, yts, \
                    l1_ratio, alphas, n_alphas, \
                    eps, fit_intercept, copy_x, max_iter, \
                    tol, cv, n_jobs, verbose, positive, selection, random_seed, id_pop, id_ind,\
                    gene_out_tr, gene_out_val, gene_out_ts
                
                # 
                results = gp_evaluate_linear_regression_crossfold(args)
                
                # Assign results
                yp_tr =       copy.deepcopy(results[0])
                yp_val =      copy.deepcopy(results[1])
                yp_ts =       copy.deepcopy(results[2])
                loss_tr =       copy.deepcopy(results[3])
                loss_val =      copy.deepcopy(results[4])
                loss_ts =       copy.deepcopy(results[5])
                weight_genes =  copy.deepcopy(results[6])
                fit_tr =        copy.deepcopy(results[3])
                fit_val =       copy.deepcopy(results[4])
                fit_ts =        copy.deepcopy(results[5])
                
                if yval is not None:
                    obj = copy.deepcopy(results[4] + results[3] + np.abs(results[4] - results[3])  + np.mean(gene_penalty_tr) +  np.mean(gene_penalty_val))
                else:
                    obj = copy.deepcopy(results[3] + np.mean(gene_penalty_tr))
            
            objective.append(fit_tr)
            
            
            res_indiv.append([copy.deepcopy(gene_out_tr),
                              copy.deepcopy(gene_out_val),
                              copy.deepcopy(gene_out_ts),
                              copy.deepcopy(gene_penalty_tr),
                              copy.deepcopy(gene_penalty_val),
                              copy.deepcopy(gene_penalty_ts),
                              copy.deepcopy(loss_tr),
                              copy.deepcopy(loss_val),
                              copy.deepcopy(loss_ts),
                              copy.deepcopy(fit_tr),
                              copy.deepcopy(fit_val),
                              copy.deepcopy(fit_ts),
                              copy.deepcopy(np.mean(gene_penalty_tr)),
                              copy.deepcopy(np.mean(gene_penalty_val)) if yval is not None else None,
                              copy.deepcopy(np.mean(gene_penalty_ts)) if yts is not None else None,
                              copy.deepcopy(yp_tr),
                              copy.deepcopy(yp_val),
                              copy.deepcopy(yp_ts),
                              copy.deepcopy(depth),
                              copy.deepcopy(num_nodes),
                              copy.deepcopy(weight_genes),
                              copy.deepcopy(complexities_isolated),
                              copy.deepcopy(obj)])
        
        sorted_idx_obj = list(np.argsort(np.array(objective) if gp.config['runcontrol']['minimisation'] else -np.array(objective)))
        
        
        for id_ind in range(pop_size):
            gene_out_tr, gene_out_val, gene_out_ts, gene_penalty_tr, gene_penalty_val, gene_penalty_ts, loss_tr, \
                loss_val, loss_ts, fit_tr, fit_val, fit_ts, pen_tr, pen_val, pen_ts, yp_tr, yp_val, yp_ts, depth, \
                num_nodes, weight_genes, complexities_isolated, obj = res_indiv[id_ind] #res_indiv[sorted_idx_obj[id_ind]]
            
            # Assign the parameters
            gp.individuals['gene_output']['train'][id_pop][id_ind] =                copy.deepcopy(gene_out_tr)
            gp.individuals['gene_output']['validation'][id_pop][id_ind] =           copy.deepcopy(gene_out_val)
            gp.individuals['gene_output']['test'][id_pop][id_ind] =                 copy.deepcopy(gene_out_ts)
            gp.individuals['gene_penalty']['train'][id_pop][id_ind] =               copy.deepcopy(gene_penalty_tr)
            gp.individuals['gene_penalty']['validation'][id_pop][id_ind] =          copy.deepcopy(gene_penalty_val)
            gp.individuals['gene_penalty']['test'][id_pop][id_ind] =                copy.deepcopy(gene_penalty_ts)
            gp.individuals['loss']['isolated']['train'][id_ind, id_pop] =           copy.deepcopy(loss_tr)
            gp.individuals['loss']['isolated']['validation'][id_ind, id_pop] =      copy.deepcopy(loss_val)
            gp.individuals['loss']['isolated']['test'][id_ind, id_pop] =            copy.deepcopy(loss_ts)
            gp.individuals['fitness']['isolated']['train'][id_ind, id_pop] =        copy.deepcopy(fit_tr)
            gp.individuals['fitness']['isolated']['validation'][id_ind, id_pop] =   copy.deepcopy(fit_val)
            gp.individuals['fitness']['isolated']['test'][id_ind, id_pop] =         copy.deepcopy(fit_ts)
            gp.individuals['penalty']['isolated']['train'][id_ind, id_pop] =        copy.deepcopy(pen_tr)
            gp.individuals['penalty']['isolated']['validation'][id_ind, id_pop] =   copy.deepcopy(pen_val)
            gp.individuals['penalty']['isolated']['test'][id_ind, id_pop] =         copy.deepcopy(pen_ts)
            gp.individuals['yp']['isolated']['train'][id_pop][id_ind] =             copy.deepcopy(yp_tr)
            gp.individuals['yp']['isolated']['validation'][id_pop][id_ind] =        copy.deepcopy(yp_val)
            gp.individuals['yp']['isolated']['test'][id_pop][id_ind] =              copy.deepcopy(yp_ts)
            gp.individuals['depth']['isolated'][id_pop][id_ind] =                   copy.deepcopy(depth)
            gp.individuals['num_nodes']['isolated'][id_pop][id_ind] =               copy.deepcopy(num_nodes)
            gp.individuals['weight_genes'][id_pop][id_ind] =                        copy.deepcopy(weight_genes) 
            gp.individuals['complexity']['isolated'][id_ind, id_pop] =              copy.deepcopy(complexities_isolated)
            gp.individuals['objective']['isolated'][id_ind, id_pop] =               copy.deepcopy(obj)
            # gp.population[id_pop][id_ind] = copy.deepcopy(popgp[id_pop][sorted_idx_obj[id_ind]])
            
        
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
    pen_en_tr =         copy.deepcopy(results_en[21])
    pen_en_val =        copy.deepcopy(results_en[22])
    pen_en_ts =         copy.deepcopy(results_en[23])
    
    if yval is not None:
        obj_ens = copy.deepcopy(results_en[19] + results_en[18] + np.abs(results_en[19] - results_en[18]) + results_en[21] + results_en[22])
    else:
        obj_ens = copy.deepcopy(results_en[18] + np.mean(results_en[21]))
   
    
    # Assigning the values
    gp.individuals['ensemble_weight'] =                         copy.deepcopy(en_weight)
    gp.individuals['ensemble_idx'] =                            copy.deepcopy(en_idx)
    gp.individuals['complexity']['ensemble'] =                  copy.deepcopy(complexity_en)
    gp.individuals['fitness']['ensemble']['train'] =            copy.deepcopy(fit_en_tr)
    gp.individuals['fitness']['ensemble']['validation'] =       copy.deepcopy(fit_en_val)
    gp.individuals['fitness']['ensemble']['test'] =             copy.deepcopy(fit_en_ts)
    gp.individuals['penalty']['ensemble']['train'] =            copy.deepcopy(pen_en_tr)
    gp.individuals['penalty']['ensemble']['validation'] =       copy.deepcopy(pen_en_val)
    gp.individuals['penalty']['ensemble']['test'] =             copy.deepcopy(pen_en_ts)
    gp.individuals['loss']['ensemble']['train'] =               copy.deepcopy(loss_en_tr)
    gp.individuals['loss']['ensemble']['validation'] =          copy.deepcopy(loss_en_val)
    gp.individuals['loss']['ensemble']['test'] =                copy.deepcopy(loss_en_ts)
    gp.individuals['yp']['ensemble']['train'] =                 copy.deepcopy(yp_en_tr)
    gp.individuals['yp']['ensemble']['validation'] =            copy.deepcopy(yp_en_val)
    gp.individuals['yp']['ensemble']['test'] =                  copy.deepcopy(yp_en_ts)
    gp.individuals['num_nodes']['ensemble'] =                   copy.deepcopy(num_nodes_en)
    gp.individuals['depth']['ensemble'] =                       copy.deepcopy(depth_en)
    gp.individuals['objective']['ensemble'] =                   copy.deepcopy(obj_ens)
    gp.individuals['all_ensemble']['idx'] =                     copy.deepcopy(id_ens)
    gp.individuals['all_ensemble']['fitness']['train'] =        copy.deepcopy(fit_ens_tr)
    gp.individuals['all_ensemble']['fitness']['validation'] =   copy.deepcopy(fit_ens_val)
    gp.individuals['all_ensemble']['fitness']['test'] =         copy.deepcopy(fit_ens_ts)
    
    for id_pop in range(num_pop):
        gp.individuals['rank']['complexity']['isolated'][:, id_pop] = np.argsort(gp.individuals['complexity']['isolated'][:, id_pop]) 
    
    gp.individuals['rank']['complexity']['ensemble'] = np.argsort(gp.individuals['complexity']['ensemble'])
    
    if gp.config['runcontrol']['minimisation']:
        gp.individuals['rank']['objective']['ensemble'] = np.argsort(gp.individuals['objective']['ensemble'])
        gp.individuals['rank']['fitness']['ensemble']['train'] = np.argsort(gp.individuals['fitness']['ensemble']['train'])
        if xval is not None:
            gp.individuals['rank']['fitness']['ensemble']['validation'] = np.argsort(gp.individuals['fitness']['ensemble']['validation'])
        if xts is not None:
            gp.individuals['rank']['fitness']['ensemble']['test'] = np.argsort(gp.individuals['fitness']['ensemble']['test'])
        
        for id_pop in range(num_pop):
            gp.individuals['rank']['objective']['isolated'][id_pop] = np.argsort(gp.individuals['objective']['isolated'][:, id_pop])
            gp.individuals['rank']['fitness']['isolated']['train'][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['train'][:, id_pop])
            if xval is not None:
                gp.individuals['rank']['fitness']['isolated']['validation'][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['validation'][:, id_pop])
            if xts is not None:
                gp.individuals['rank']['fitness']['isolated']['test'][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['test'][:, id_pop])
    else:
        gp.individuals['rank']['objective']['ensemble'] = np.argsort(-gp.individuals['objective']['ensemble'])
        gp.individuals['rank']['fitness']['ensemble']['train'] = np.argsort(-gp.individuals['fitness']['ensemble']['train'])
        if xval is not None:
            gp.individuals['rank']['fitness']['ensemble']['validation'] = np.argsort(-gp.individuals['fitness']['ensemble']['validation'])
        if xts is not None:
            gp.individuals['rank']['fitness']['ensemble']['test'] = np.argsort(-gp.individuals['fitness']['ensemble']['test'])
        
        for id_pop in range(num_pop):
            gp.individuals['rank']['objective']['isolated'][id_pop] = np.argsort(-gp.individuals['objective']['isolated'][:, id_pop])
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
        gp.track['penalty']['isolated']['train'][gen] =             copy.deepcopy(gp.individuals['penalty']['isolated']['train'])
        gp.track['penalty']['isolated']['validation'][gen] =        copy.deepcopy(gp.individuals['penalty']['isolated']['validation'])
        gp.track['penalty']['isolated']['test'][gen] =              copy.deepcopy(gp.individuals['penalty']['isolated']['test'])
        gp.track['penalty']['ensemble']['train'][gen] =             copy.deepcopy(gp.individuals['penalty']['ensemble']['train'])
        gp.track['penalty']['ensemble']['validation'][gen] =        copy.deepcopy(gp.individuals['penalty']['ensemble']['validation'])
        gp.track['penalty']['ensemble']['test'][gen] =              copy.deepcopy(gp.individuals['penalty']['ensemble']['test'])
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
        gp.track['weight_genes'][gen] =                             copy.deepcopy(gp.individuals['weight_genes'])
        gp.track['ensemble_weight'][gen] =                          copy.deepcopy(gp.individuals['ensemble_weight'])
        gp.track['yp']['isolated']['train'][gen] =                  copy.deepcopy(gp.individuals['yp']['isolated']['train'])
        gp.track['yp']['isolated']['validation'][gen] =             copy.deepcopy(gp.individuals['yp']['isolated']['validation'])
        gp.track['yp']['isolated']['test'][gen] =                   copy.deepcopy(gp.individuals['yp']['isolated']['test'])
        gp.track['yp']['ensemble']['train'][gen] =                  copy.deepcopy(gp.individuals['yp']['ensemble']['train'])
        gp.track['yp']['ensemble']['validation'][gen] =             copy.deepcopy(gp.individuals['yp']['ensemble']['validation'])
        gp.track['yp']['ensemble']['test'][gen] =                   copy.deepcopy(gp.individuals['yp']['ensemble']['test'])
        gp.track['objective']['isolated'][gen] =                    copy.deepcopy(gp.individuals['objective']['isolated'])
        gp.track['objective']['ensemble'][gen] =                    copy.deepcopy(gp.individuals['objective']['ensemble'])
        
        for id_pop in range(num_pop):
            gp.track['rank']['complexity']['isolated'][gen][:, id_pop] = np.argsort(gp.individuals['complexity']['isolated'][:, id_pop]) 
        
        gp.track['rank']['complexity']['ensemble'][gen] = np.argsort(gp.individuals['complexity']['ensemble'])
        
        if gp.config['runcontrol']['minimisation']:
            gp.track['rank']['objective']['ensemble'][gen] = np.argsort(gp.individuals['objective']['ensemble'])
            gp.track['rank']['fitness']['ensemble']['train'][gen] = np.argsort(gp.individuals['fitness']['ensemble']['train'])
            if xval is not None:
                gp.track['rank']['fitness']['ensemble']['validation'][gen] = np.argsort(gp.individuals['fitness']['ensemble']['validation'])
            if xts is not None:
                gp.track['rank']['fitness']['ensemble']['test'][gen] = np.argsort(gp.individuals['fitness']['ensemble']['test'])
            
            for id_pop in range(num_pop):
                gp.track['rank']['objective']['isolated'][gen][id_pop] = np.argsort(gp.individuals['objective']['isolated'][:, id_pop])
                gp.track['rank']['fitness']['isolated']['train'][gen][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['train'][:, id_pop])
                if xval is not None:
                    gp.track['rank']['fitness']['isolated']['validation'][gen][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['validation'][:, id_pop])
                if xts is not None:
                    gp.track['rank']['fitness']['isolated']['test'][gen][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['test'][:, id_pop])
        else:
            gp.track['rank']['objective']['ensemble'][gen] = np.argsort(-gp.individuals['objective']['ensemble'])
            gp.track['rank']['fitness']['ensemble']['train'][gen] = np.argsort(-gp.individuals['fitness']['ensemble']['train'])
            if xval is not None:
                gp.track['rank']['fitness']['ensemble']['validation'][gen] = np.argsort(-gp.individuals['fitness']['ensemble']['validation'])
            if xts is not None:
                gp.track['rank']['fitness']['ensemble']['test'][gen] = np.argsort(-gp.individuals['fitness']['ensemble']['test'])
            
            for id_pop in range(num_pop):
                gp.track['rank']['objective']['isolated'][gen][id_pop] = np.argsort(-gp.individuals['objective']['isolated'][:, id_pop])
                gp.track['rank']['fitness']['isolated']['train'][gen][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['train'][:, id_pop])
                if xval is not None:
                    gp.track['rank']['fitness']['isolated']['validation'][gen][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['validation'][:, id_pop])
                if xts is not None:
                    gp.track['rank']['fitness']['isolated']['test'][gen][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['test'][:, id_pop])
    
