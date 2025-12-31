# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
from scipy.optimize import minimize
import copy

def logloss(prob, Y):
    prob = np.clip(prob, 1e-12, 1 - 1e-12)
    log_likelihood = -np.sum(Y * np.log(prob), axis=1)
    loss = np.mean(log_likelihood)
    return loss
def generate_combinations(object_list):
    # Get the number of elements in each sublist of the object list
    num_object = [len(obj) for obj in object_list]
    # Get the number of places
    num_place = len(object_list)
    # Calculate the total number of combinations
    num_choice = 1
    for i in range(num_place):
        num_choice *= num_object[i]
    num_combination = num_choice
    # Calculate the number of choices for each place
    num_choice_place = [0] * num_place
    num_choice_place[0] = num_choice // num_object[0]
    for i in range(1, num_place):
        num_choice_place[i] = num_choice_place[i - 1] // num_object[i]
    # Initialize the combination list
    combination = np.zeros((num_choice, num_place), dtype=int)
    # Fill the combination list
    for i in range(num_choice):
        for j in range(num_place):
            idx = int(np.ceil((i / num_choice_place[j]) % num_object[j]))
            combination[i, j] = object_list[j][idx - 1]  # Subtract 1 for 0-based indexing in Python
    return num_combination, combination.tolist()

def ensemble_task(comb_ens, prob_comb_tr, prob_comb_val, prob_comb_ts, ybin_tr, ybin_val, ybin_ts):
    # Define weight normalization function
    def normal_weight(weights1):
        normalized_weights = np.zeros_like(weights1)
        for j in range(prob_comb_tr[0].shape[1]):  # Iterate over classes
            start_idx = j * len(prob_comb_tr)
            end_idx = (j + 1) * len(prob_comb_tr)
            class_weights = weights1[start_idx:end_idx]
            sum_class_weights = np.sum(class_weights)
            if sum_class_weights != 0:
                normalized_weights[start_idx:end_idx] = class_weights / sum_class_weights
            else:
                normalized_weights[start_idx:end_idx] = np.zeros_like(class_weights)
        return normalized_weights
    
    # Define the loss function for optimizing ensemble weights
    def loss_function(weights1):
        combined_prob = np.zeros_like(prob_comb_tr[0])
        normalized_weights = normal_weight(weights1)
        for j in range(prob_comb_tr[0].shape[1]):
            for k in range(len(prob_comb_tr)):
                combined_prob[:, j] += prob_comb_tr[k][:, j] * normalized_weights[j * len(prob_comb_tr) + k]
        mse_loss = np.mean(np.sum(np.square(ybin_tr - combined_prob), axis = 1))
        return mse_loss
    
    # Initial guess for weights (even distribution)
    initial_weights = np.ones(len(prob_comb_tr) * prob_comb_tr[0].shape[1]) / len(prob_comb_tr)
    # Bounds: Weights between 0, and 1
    bounds = [(0, 1) for _ in range(len(initial_weights))]
    # Optimize the weights
    result = minimize(loss_function, initial_weights, method='COBYLA', bounds=bounds)
    # ordering the weights for better use
    normalized_weights = normal_weight(result.x)
    prob_ens_tr = np.zeros_like(prob_comb_tr[0])
    prob_ens_val = np.zeros_like(prob_comb_val[0]) if ybin_val is not None else None
    prob_ens_ts = np.zeros_like(prob_comb_ts[0]) if ybin_ts is not None else None
    weights = []
    for k in range(len(prob_comb_tr)):
        weights.append([])
        for j in range(prob_comb_tr[0].shape[1]):
            weights[k].append(normalized_weights[j * len(prob_comb_tr) + k])
            prob_ens_tr [:, j] += prob_comb_tr[k][:, j] * normalized_weights[j * len(prob_comb_tr) + k]
            if ybin_val is not None:
                prob_ens_val [:, j] += prob_comb_val[k][:, j] * normalized_weights[j * len(prob_comb_val) + k]
            if ybin_ts is not None:
                prob_ens_ts [:, j] += prob_comb_ts[k][:, j] * normalized_weights[j * len(prob_comb_ts) + k]
    # Cross Entropy of ensemble
    loss_tr = logloss(prob_ens_tr, ybin_tr)
    loss_val = logloss(prob_ens_val, ybin_val) if ybin_val is not None else None
    loss_ts = logloss(prob_ens_ts, ybin_ts) if ybin_ts is not None else None
    idx_en = comb_ens
    return idx_en, weights, prob_ens_tr, prob_ens_val, prob_ens_ts, loss_tr, loss_val, loss_ts

def gp_evaluate_ensemble_ord(gp):
    """
    Calculate the best ensemble weights for each individual in the id_pop=0 population by combining them
    with every combination of individuals from other populations.

    Args:
    gp (object): The GP object containing individuals' probabilities and labels.
    """
    pop_size = gp.config['runcontrol']['pop_size']
    num_pop = gp.config['runcontrol']['num_pop']
    depth = copy.deepcopy(gp.individuals['depth']['isolated'])
    num_nodes = copy.deepcopy(gp.individuals['num_nodes']['isolated'])
    ybin_tr = gp.userdata['ybinarytrain']
    ybin_val = gp.userdata['ybinaryval']
    ybin_ts = gp.userdata['ybinarytest']
    prob_tr = copy.deepcopy(gp.individuals['prob']['isolated']['train'])
    prob_val = copy.deepcopy(gp.individuals['prob']['isolated']['validation'])
    prob_ts = copy.deepcopy(gp.individuals['prob']['isolated']['test'])
    complexity = copy.deepcopy(gp.individuals['complexity']['isolated'])
    
    # Determine Combinations of Ensembles
    num_ens_comb, ens_comb = generate_combinations([list(np.arange(0, pop_size)) for _ in range(num_pop)])
    
    weight_all = [None for _ in range(num_ens_comb)]
    prob_en_tr_all = [None for _ in range(num_ens_comb)]
    prob_en_val_all = [None for _ in range(num_ens_comb)] if ybin_val is not None else None
    prob_en_ts_all = [None for _ in range(num_ens_comb)] if ybin_ts is not None else None
    fit_ens_tr = np.full((num_ens_comb), np.inf)
    fit_ens_val = np.full((num_ens_comb), np.inf) if ybin_val is not None else None
    fit_ens_ts = np.full((num_ens_comb), np.inf) if ybin_ts is not None else None
    # Ensembling task one-by-one
    for ens in range(num_ens_comb):
        prob_comb_tr = []
        prob_comb_val = [] if ybin_val is not None else None
        prob_comb_ts = [] if ybin_ts is not None else None
        for id_pop in range(len(ens_comb[ens])):
            prob_comb_tr.append(copy.deepcopy(prob_tr[id_pop][ens_comb[ens][id_pop]]))
            prob_comb_val.append(copy.deepcopy(prob_val[id_pop][ens_comb[ens][id_pop]])) if ybin_val is not None else None
            prob_comb_ts.append(copy.deepcopy(prob_ts[id_pop][ens_comb[ens][id_pop]])) if ybin_ts is not None else None
            
        results_task = ensemble_task(ens_comb[ens], prob_comb_tr, prob_comb_val, prob_comb_ts,\
            ybin_tr, ybin_val, ybin_ts)
        
        # _ = results_task[0]
        weight_all[ens] = results_task[1]
        prob_en_tr_all[ens] = results_task[2]
        if ybin_val is not None:
            prob_en_val_all[ens] = results_task[3]
        if ybin_ts is not None:
            prob_en_ts_all[ens] = results_task[4]
        fit_ens_tr[ens] = results_task[5]
        if ybin_val is not None:
            fit_ens_val[ens] = results_task[6]
        if ybin_ts is not None:
            fit_ens_ts[ens] = results_task[7]
        
    # Finding the best ensemble for each individual in id_pop = 0
    id_ens_best = [None for _ in range(pop_size)]
    fit_best = np.full((pop_size), np.inf)
    for en_id in range(num_ens_comb):
        if fit_ens_tr[en_id] < fit_best[ens_comb[en_id][0]]:
            fit_best[ens_comb[en_id][0]] = fit_ens_tr[en_id]
            id_ens_best[ens_comb[en_id][0]] = en_id
    
    # Assign the best results 
    en_weight = [None for _ in range(pop_size)]
    en_idx = np.zeros((pop_size, num_pop), dtype=int)
    complexity_en = np.zeros((pop_size))
    prob_en_tr = [None for _ in range(pop_size)]
    prob_en_val = [None for _ in range(pop_size)] if ybin_val is not None else None
    prob_en_ts = [None for _ in range(pop_size)] if ybin_ts is not None else None
    loss_en_tr = np.zeros((pop_size))
    loss_en_val = np.zeros((pop_size)) if ybin_val is not None else None
    loss_en_ts = np.zeros((pop_size)) if ybin_ts is not None else None
    fit_en_tr = np.full(pop_size, np.inf)
    fit_en_val = np.full(pop_size, np.inf) if ybin_val is not None else None
    fit_en_ts = np.full(pop_size, np.inf) if ybin_ts is not None else None
    yp_en_tr = [None for _ in range(pop_size)]
    yp_en_val = [None for _ in range(pop_size)] if ybin_val is not None else None
    yp_en_ts = [None for _ in range(pop_size)] if ybin_ts is not None else None
    depth_en = [None for _ in range(pop_size)]
    num_nodes_en = [None for _ in range(pop_size)]
    id_ens = np.array(ens_comb, dtype = int)
    
    for best_id in range(pop_size):
        en_weight[best_id] = copy.deepcopy(weight_all[id_ens_best[best_id]])
        prob_en_tr[best_id] = copy.deepcopy(prob_en_tr_all[id_ens_best[best_id]])
        if ybin_val is not None:
            prob_en_val[best_id] = copy.deepcopy(prob_en_val_all[id_ens_best[best_id]])
        if ybin_ts is not None:
            prob_en_ts[best_id] = copy.deepcopy(prob_en_ts_all[id_ens_best[best_id]])
        loss_en_tr[best_id] = copy.deepcopy(fit_ens_tr[id_ens_best[best_id]])
        if ybin_val is not None:
            loss_en_val[best_id] = copy.deepcopy(fit_ens_val[id_ens_best[best_id]])
        if ybin_ts is not None:
            loss_en_ts[best_id] = copy.deepcopy(fit_ens_ts[id_ens_best[best_id]])
        fit_en_tr[best_id] = copy.deepcopy(fit_ens_tr[id_ens_best[best_id]])
        if ybin_val is not None:
            fit_en_val[best_id] = copy.deepcopy(fit_ens_val[id_ens_best[best_id]])
        if ybin_ts is not None:
            fit_en_ts[best_id] = copy.deepcopy(fit_ens_ts[id_ens_best[best_id]])
        yp_en_tr[best_id] = np.argmax(prob_en_tr[best_id], axis=1)
        if ybin_val is not None:
            yp_en_val[best_id] = np.argmax(prob_en_val[best_id], axis=1)
        if ybin_ts is not None:
            yp_en_ts[best_id] = np.argmax(prob_en_ts[best_id], axis=1)
        
        depth_en[best_id] = []
        num_nodes_en[best_id] = []
        for id_pop in range(num_pop):
            en_idx[best_id, id_pop] = copy.deepcopy(ens_comb[id_ens_best[best_id]][id_pop])
            complexity_en[best_id] += complexity[en_idx[best_id, id_pop], id_pop]
            depth_en[best_id].append(copy.deepcopy(depth[id_pop][en_idx[best_id, id_pop]]))
            num_nodes_en[best_id].append(copy.deepcopy(num_nodes[id_pop][en_idx[best_id, id_pop]]))
            
    # Store in the results list
    results_en = [
        en_weight,
        en_idx,
        complexity_en,
        prob_en_tr,
        prob_en_val,
        prob_en_ts,
        loss_en_tr,
        loss_en_val,
        loss_en_ts,
        fit_en_tr,
        fit_en_val,
        fit_en_ts,
        yp_en_tr,
        yp_en_val,
        yp_en_ts,
        depth_en,
        num_nodes_en,
        id_ens,
        fit_ens_tr,
        fit_ens_val,
        fit_ens_ts,
    ]
    return results_en
            
            
            