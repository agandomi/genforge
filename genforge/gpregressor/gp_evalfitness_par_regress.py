# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import traceback

from .gp_evaluate_tree_regress import gp_evaluate_tree
from .gp_getdepth import gp_getdepth
from .gp_getcomplexity import gp_getcomplexity
from .gp_getnumnodes import gp_getnumnodes
from .gp_evaluate_linear_regression_crossfold import gp_evaluate_linear_regression_crossfold

# Prefer parallel ensemble evaluation if provided; fall back to ordinary
try:
    from .gp_evaluate_ensemble_par_regress import gp_evaluate_ensemble_par as _eval_ensemble
except Exception:
    from .gp_evaluate_ensemble_ord_regress import gp_evaluate_ensemble_ord as _eval_ensemble


def _evaluate_individual_batch(args):
    """
    Worker: evaluate a batch of individuals for a single population (regression).
    Returns a list of tuples with all artifacts needed to fill gp.individuals.
    """
    (
        id_pop, id_ind_range,
        xtr, xval, xts, ytr, yval, yts, ztr, zval, zts,
        function_map_all, popgp, complexity_measure,
        # linear regression head (per-pop):
        l1_ratio_all, alphas_all, n_alphas_all, \
        eps_all, fit_intercept_all, copy_x_all, max_iter_all, \
        tol_all, cv_all, n_jobs_all, verbose_all, positive_all, selection_all, random_seed_all
    ) = args

    function_map = function_map_all[id_pop]
    l1_ratio = l1_ratio_all[id_pop]
    alphas = alphas_all[id_pop]
    n_alphas = n_alphas_all[id_pop]
    eps = eps_all[id_pop]
    fit_intercept = fit_intercept_all[id_pop]
    copy_x = copy_x_all[id_pop]
    max_iter = max_iter_all[id_pop]
    tol = tol_all[id_pop]
    cv = cv_all[id_pop]
    n_jobs = n_jobs_all[id_pop]
    verbose = verbose_all[id_pop]
    positive = positive_all[id_pop]
    selection = selection_all[id_pop]
    random_seed = random_seed_all
    

    results_batch = []

    for id_ind in id_ind_range:
        ind = popgp[id_pop][id_ind]

        # per-gene outputs and penalties
        G = len(ind)
        gene_out_tr = np.zeros((xtr.shape[0], G))
        gene_out_val = np.zeros((xval.shape[0], G)) if yval is not None else None
        gene_out_ts = np.zeros((xts.shape[0], G)) if yts is not None else None

        gene_penalty_tr = np.zeros((xtr.shape[0], G))
        gene_penalty_val = np.zeros((xval.shape[0], G)) if yval is not None else None
        gene_penalty_ts = np.zeros((xts.shape[0], G)) if yts is not None else None

        depth = np.zeros(G)
        num_nodes = np.zeros(G)
        complexities_isolated = 0

        for id_gene in range(G):
            # evaluate gene to get output and penalty on each split
            go_tr, gp_tr = gp_evaluate_tree(ind[id_gene], xtr, ztr, function_map)
            if yval is not None:
                go_val, gp_val = gp_evaluate_tree(ind[id_gene], xval, zval, function_map)
            if yts is not None:
                go_ts, gp_ts = gp_evaluate_tree(ind[id_gene], xts, zts, function_map)

            gene_out_tr[:, id_gene] = go_tr
            if yval is not None:
                gene_out_val[:, id_gene] = go_val
                
            if yts is not None:
                gene_out_ts[:, id_gene] = go_ts

            gene_penalty_tr[:, id_gene] = gp_tr
            if yval is not None:
                gene_penalty_val[:, id_gene] = gp_val
            if yts is not None:
                gene_penalty_ts[:, id_gene] = gp_ts

            depth[id_gene] = gp_getdepth(ind[id_gene])
            num_nodes[id_gene] = gp_getnumnodes(ind[id_gene])

            if complexity_measure == 1:
                complexities_isolated += gp_getcomplexity(ind[id_gene])
            else:
                complexities_isolated += gp_getnumnodes(ind[id_gene])

        # Train/evaluate linear regression head (cross-fold) on gene outputs
        args_linreg = (
            ytr, yval, yts,
            l1_ratio, alphas, n_alphas, \
            eps, fit_intercept, copy_x, max_iter, \
            tol, cv, n_jobs, verbose, positive, selection, random_seed, id_pop, id_ind,
            gene_out_tr, gene_out_val, gene_out_ts
        )
        results = gp_evaluate_linear_regression_crossfold(args_linreg)

        # Unpack lr results
        yp_tr, yp_val, yp_ts = results[0], results[1], results[2]
        loss_tr, loss_val, loss_ts = results[3], results[4], results[5]
        weight_genes = results[6]

        # fitness == loss for regression (as in ordinary version)
        fit_tr, fit_val, fit_ts = loss_tr, loss_val, loss_ts

        # objective as in ordinary version
        pen_tr_mean = float(np.mean(gene_penalty_tr)) if gene_penalty_tr.size else 0.0
        if yval is not None:
            pen_val_mean = float(np.mean(gene_penalty_val)) if gene_penalty_val.size else 0.0
        else:
            pen_val_mean = None
        
        if yval is not None:
            obj = float(loss_val + loss_tr + np.abs(loss_val - loss_tr) + pen_tr_mean + pen_val_mean)
        else:
            obj = float(loss_tr + pen_tr_mean)

        results_batch.append(
            (
                id_pop, id_ind,
                gene_out_tr, gene_out_val, gene_out_ts,
                gene_penalty_tr, gene_penalty_val, gene_penalty_ts,
                loss_tr, loss_val, loss_ts,
                fit_tr, fit_val, fit_ts,
                yp_tr, yp_val, yp_ts,
                depth, num_nodes, weight_genes,
                complexities_isolated, pen_tr_mean, pen_val_mean, obj
            )
        )

    return results_batch


def gp_evalfitness_par(gp):
    """
    Parallel evaluation of individuals for GP Regressor (structurally aligned with:
      - ordinary regressor evaluator, and
      - classifier parallel evaluator).

    Populations/individuals that are present in cache are copied directly; others are
    batched and evaluated across processes. Ensemble evaluation uses a parallel
    version if available, otherwise falls back to ordinary ensemble evaluation.
    """
    gen = gp.state['generation']
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    popgp = copy.deepcopy(gp.population)
    function_map = gp.config['nodes']['functions']['function']
    complexity_measure = gp.config['fitness']['complexityMeasure']

    # linear regression head params (per-pop)
    l1_ratio = gp.config['linregression']['l1_ratio']
    alphas = gp.config['linregression']['alphas']
    n_alphas = gp.config['linregression']['n_alphas']
    eps = gp.config['linregression']['eps']
    fit_intercept = gp.config['linregression']['fit_intercept']
    copy_x = gp.config['linregression']['copy_x']
    max_iter = gp.config['linregression']['max_iter']
    tol = gp.config['linregression']['tol']
    cv = gp.config['linregression']['cv']
    n_jobs = gp.config['linregression']['n_jobs']
    verbose = gp.config['linregression']['verbose']
    positive = gp.config['linregression']['positive']
    selection = gp.config['linregression']['selection']
    random_seed = gp.config['runcontrol']['random_state']

    # data
    xtr, xval, xts = gp.userdata['xtrain'], gp.userdata['xval'], gp.userdata['xtest']
    ytr, yval, yts = gp.userdata['ytrain'], gp.userdata['yval'], gp.userdata['ytest']
    ztr, zval, zts = gp.state['ztrain'], gp.state['zval'], gp.state['ztest']

    # build worklist for not-cached individuals
    id_ind_list = [[] for _ in range(num_pop)]
    for id_pop in range(num_pop):
        for id_ind in range(pop_size):
            if gp.config['runcontrol']['usecache'] and gp.cache['gene_output']['train'][id_pop][id_ind] is not None:
                # Copy from cache (keeps semantics of ordinary version)
                gp.individuals['gene_output']['train'][id_pop][id_ind] = copy.deepcopy(gp.cache['gene_output']['train'][id_pop][id_ind])
                gp.individuals['gene_output']['validation'][id_pop][id_ind] = copy.deepcopy(gp.cache['gene_output']['validation'][id_pop][id_ind])
                gp.individuals['gene_output']['test'][id_pop][id_ind] = copy.deepcopy(gp.cache['gene_output']['test'][id_pop][id_ind])

                gp.individuals['gene_penalty']['train'][id_pop][id_ind] = copy.deepcopy(gp.cache['gene_penalty']['train'][id_pop][id_ind])
                gp.individuals['gene_penalty']['validation'][id_pop][id_ind] = copy.deepcopy(gp.cache['gene_penalty']['validation'][id_pop][id_ind])
                gp.individuals['gene_penalty']['test'][id_pop][id_ind] = copy.deepcopy(gp.cache['gene_penalty']['test'][id_pop][id_ind])

                gp.individuals['loss']['isolated']['train'][id_ind, id_pop] = copy.deepcopy(gp.cache['loss']['isolated']['train'][id_pop][id_ind])
                gp.individuals['loss']['isolated']['validation'][id_ind, id_pop] = copy.deepcopy(gp.cache['loss']['isolated']['validation'][id_pop][id_ind])
                gp.individuals['loss']['isolated']['test'][id_ind, id_pop] = copy.deepcopy(gp.cache['loss']['isolated']['test'][id_pop][id_ind])

                gp.individuals['yp']['isolated']['train'][id_pop][id_ind] = copy.deepcopy(gp.cache['yp']['isolated']['train'][id_pop][id_ind])
                gp.individuals['yp']['isolated']['validation'][id_pop][id_ind] = copy.deepcopy(gp.cache['yp']['isolated']['validation'][id_pop][id_ind])
                gp.individuals['yp']['isolated']['test'][id_pop][id_ind] = copy.deepcopy(gp.cache['yp']['isolated']['test'][id_pop][id_ind])

                gp.individuals['fitness']['isolated']['train'][id_ind, id_pop] = copy.deepcopy(gp.cache['fitness']['isolated']['train'][id_pop][id_ind])
                gp.individuals['fitness']['isolated']['validation'][id_ind, id_pop] = copy.deepcopy(gp.cache['fitness']['isolated']['validation'][id_pop][id_ind])
                gp.individuals['fitness']['isolated']['test'][id_ind, id_pop] = copy.deepcopy(gp.cache['fitness']['isolated']['test'][id_pop][id_ind])

                gp.individuals['depth']['isolated'][id_pop][id_ind] = copy.deepcopy(gp.cache['depth']['isolated'][id_pop][id_ind])
                gp.individuals['num_nodes']['isolated'][id_pop][id_ind] = copy.deepcopy(gp.cache['num_nodes']['isolated'][id_pop][id_ind])
                gp.individuals['weight_genes'][id_pop][id_ind] = copy.deepcopy(gp.cache['weight_genes'][id_pop][id_ind])
                gp.individuals['complexity']['isolated'][id_ind, id_pop] = copy.deepcopy(gp.cache['complexity']['isolated'][id_pop][id_ind])
                gp.individuals['objective']['isolated'][id_ind, id_pop] = copy.deepcopy(gp.cache['objective']['isolated'][id_pop][id_ind])
            else:
                id_ind_list[id_pop].append(id_ind)

    # Parallel processing of not-cached individuals
    max_workers = gp.config['runcontrol']['parallel']['n_jobs']
    batch_job = gp.config['runcontrol']['parallel']['batch_job']

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for id_pop in range(num_pop):
            if not id_ind_list[id_pop]:
                continue
            id_inds = np.array(id_ind_list[id_pop], dtype=int)
            for i in range(0, len(id_inds), batch_job):
                id_ind_range = list(id_inds[i: i + batch_job])
                
                l1_ratio = gp.config['linregression']['l1_ratio']
                alphas = gp.config['linregression']['alphas']
                n_alphas = gp.config['linregression']['n_alphas']
                eps = gp.config['linregression']['eps']
                fit_intercept = gp.config['linregression']['fit_intercept']
                copy_x = gp.config['linregression']['copy_x']
                max_iter = gp.config['linregression']['max_iter']
                tol = gp.config['linregression']['tol']
                cv = gp.config['linregression']['cv']
                n_jobs = gp.config['linregression']['n_jobs']
                verbose = gp.config['linregression']['verbose']
                positive = gp.config['linregression']['positive']
                selection = gp.config['linregression']['selection']
                random_seed = gp.config['runcontrol']['random_state']
                
                worker_args = (
                    id_pop, id_ind_range,
                    xtr, xval, xts, ytr, yval, yts, ztr, zval, zts,
                    function_map, popgp, complexity_measure,
                    l1_ratio, alphas, n_alphas,
                    eps, fit_intercept, copy_x, max_iter,
                    tol, cv, n_jobs, verbose, positive, selection, random_seed
                )
                futures.append(executor.submit(_evaluate_individual_batch, worker_args))

        for fut in as_completed(futures):
            try:
                results_batch = fut.result()
                for result in results_batch:
                    (
                        id_pop, id_ind,
                        gene_out_tr, gene_out_val, gene_out_ts,
                        gene_penalty_tr, gene_penalty_val, gene_penalty_ts,
                        loss_tr, loss_val, loss_ts,
                        fit_tr, fit_val, fit_ts,
                        yp_tr, yp_val, yp_ts,
                        depth, num_nodes, weight_genes,
                        complexities_isolated, pen_tr_mean, pen_val_mean, obj
                    ) = result
                    
                    # Write back to gp.individuals
                    gp.individuals['gene_output']['train'][id_pop][id_ind] = copy.deepcopy(gene_out_tr)
                    gp.individuals['gene_output']['validation'][id_pop][id_ind] = copy.deepcopy(gene_out_val) if gene_out_val is not None else None
                    gp.individuals['gene_output']['test'][id_pop][id_ind] = copy.deepcopy(gene_out_ts) if gene_out_ts is not None else None

                    gp.individuals['gene_penalty']['train'][id_pop][id_ind] = copy.deepcopy(gene_penalty_tr)
                    gp.individuals['gene_penalty']['validation'][id_pop][id_ind] = copy.deepcopy(gene_penalty_val) if gene_penalty_val is not None else None
                    gp.individuals['gene_penalty']['test'][id_pop][id_ind] = copy.deepcopy(gene_penalty_ts) if gene_penalty_ts is not None else None

                    gp.individuals['loss']['isolated']['train'][id_ind, id_pop] = float(loss_tr)
                    gp.individuals['loss']['isolated']['validation'][id_ind, id_pop] = float(loss_val) if loss_val is not None else None
                    gp.individuals['loss']['isolated']['test'][id_ind, id_pop] = float(loss_ts) if loss_ts is not None else None

                    gp.individuals['fitness']['isolated']['train'][id_ind, id_pop] = float(fit_tr)
                    gp.individuals['fitness']['isolated']['validation'][id_ind, id_pop] = float(fit_val) if fit_val is not None else None
                    gp.individuals['fitness']['isolated']['test'][id_ind, id_pop] = float(fit_ts) if loss_ts is not None else None

                    gp.individuals['penalty']['isolated']['train'][id_ind, id_pop] = float(pen_tr_mean)
                    gp.individuals['penalty']['isolated']['validation'][id_ind, id_pop] = float(pen_val_mean) if pen_val_mean is not None else None
                    # test penalty mean (if needed) can be added analogously
                    gp.individuals['penalty']['isolated']['test'][id_ind, id_pop] = (float(np.mean(gene_penalty_ts)) if gene_penalty_ts.size else 0.0) if gene_penalty_ts is not None else None

                    gp.individuals['yp']['isolated']['train'][id_pop][id_ind] = copy.deepcopy(yp_tr)
                    gp.individuals['yp']['isolated']['validation'][id_pop][id_ind] = copy.deepcopy(yp_val) if yp_val is not None else None
                    gp.individuals['yp']['isolated']['test'][id_pop][id_ind] = copy.deepcopy(yp_ts) if yp_ts is not None else None

                    gp.individuals['depth']['isolated'][id_pop][id_ind] = copy.deepcopy(depth)
                    gp.individuals['num_nodes']['isolated'][id_pop][id_ind] = copy.deepcopy(num_nodes)
                    gp.individuals['weight_genes'][id_pop][id_ind] = copy.deepcopy(weight_genes)
                    gp.individuals['complexity']['isolated'][id_ind, id_pop] = float(complexities_isolated)
                    gp.individuals['objective']['isolated'][id_ind, id_pop] = float(obj)

            except Exception as exc:
                print(f"[gp_evalfitness_par_regress] Worker exception: {exc}")
                traceback.print_exc()

    # ----- Evaluate all ensembles (parallel implementation preferred) -----
    results_en = _eval_ensemble(gp)

    # Unpack ensemble results (aligned with ordinary regressor)
    en_weight       = copy.deepcopy(results_en[0])
    en_idx          = copy.deepcopy(results_en[1])
    complexity_en   = copy.deepcopy(results_en[2])
    prob_en_tr      = copy.deepcopy(results_en[3])
    prob_en_val     = copy.deepcopy(results_en[4])
    prob_en_ts      = copy.deepcopy(results_en[5])
    loss_en_tr      = copy.deepcopy(results_en[6])
    loss_en_val     = copy.deepcopy(results_en[7])
    loss_en_ts      = copy.deepcopy(results_en[8])
    fit_en_tr       = copy.deepcopy(results_en[9])
    fit_en_val      = copy.deepcopy(results_en[10])
    fit_en_ts       = copy.deepcopy(results_en[11])
    yp_en_tr        = copy.deepcopy(results_en[12])
    yp_en_val       = copy.deepcopy(results_en[13])
    yp_en_ts        = copy.deepcopy(results_en[14])
    depth_en        = copy.deepcopy(results_en[15])
    num_nodes_en    = copy.deepcopy(results_en[16])
    id_ens          = copy.deepcopy(results_en[17])
    fit_ens_tr      = copy.deepcopy(results_en[18])
    fit_ens_val     = copy.deepcopy(results_en[19])
    fit_ens_ts      = copy.deepcopy(results_en[20])
    pen_en_tr       = copy.deepcopy(results_en[21])
    pen_en_val      = copy.deepcopy(results_en[22])
    pen_en_ts       = copy.deepcopy(results_en[23])

    if yval is not None:
        obj_ens = copy.deepcopy(results_en[19] + results_en[18] + np.abs(results_en[19] - results_en[18]) + results_en[21] + results_en[22])
    else:
        obj_ens = copy.deepcopy(results_en[18] + results_en[21])
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
