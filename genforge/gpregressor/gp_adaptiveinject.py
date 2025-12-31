# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
from itertools import combinations
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_predict
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
import copy
import warnings

def cross_val_mse(X, y, random_state):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = ElasticNetCV(l1_ratio=[1], cv=5, random_state=random_state, max_iter=1000)
        y_pred = cross_val_predict(model, X, y, cv=5)
        return mean_squared_error(y, y_pred)

def adaptive_gene_pruning(X_full, gene_names, y, mse_z1, epsilon=1e-4, random_state=None):
    best_X = None
    best_genes = None
    best_mse = np.inf
    for r in [3, 2, 1]:
        for idxs in combinations(range(X_full.shape[1]), r):
            X_subset = X_full[:, idxs]
            mse = cross_val_mse(X_subset, y, random_state)
            if mse <= mse_z1 + epsilon:
                return X_subset, [gene_names[i] for i in idxs]
            if mse < best_mse:
                best_mse = mse
                best_X = X_subset
                best_genes = [gene_names[i] for i in idxs]
    return best_X, best_genes

def detect_z1_z2_presence(kept_genes, z1_token, z2_token):
    z1_present = any(z1_token in gene for gene in kept_genes)
    z2_present = any(z2_token in gene for gene in kept_genes)
    return [z1_present, z2_present]

def calculate_weights(w_gene, w_ens):
    num_genes = sum(len(w) - 1 for w in w_gene)
    w_new = [0 for _ in range(num_genes + 1)]
    id_gene = 0
    for jj in range(len(w_gene)):
        w_new[-1] += (w_ens[jj] * w_gene[jj][-1]).item()
        for kk in range(len(w_gene[jj]) - 1):
            w_new[id_gene] = (w_ens[jj] * w_gene[jj][kk]).item()
            id_gene += 1
        
    w_new[-1] += w_ens[-1]
    return w_new

def build_expression(w_new, indiv_all):
    indiv_temp = []
    for jj in range(len(indiv_all)):
        if w_new[jj] != 0:
            indiv_temp.append('times([' + f'{w_new[jj]}' + '],' + indiv_all[jj] + ')')
    
    indiv_tem_final = indiv_temp[0]
    for jj in range(len(indiv_temp) - 1):
        indiv_tem_final = 'plus(' + indiv_tem_final + ',' + indiv_temp[jj + 1] + ')'
    
    indiv_tem_final = 'plus(' + indiv_tem_final + ',[' + f'{w_new[-1]}' + '])'
    return indiv_tem_final

def retrain_weights(X, y, random_state):
    """Retrain ElasticNet weights on selected gene outputs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = ElasticNetCV(
            l1_ratio=[1], cv=5, random_state=random_state,
            max_iter=1000, fit_intercept=True
        ).fit(X, y)
    w = model.coef_.astype(float)
    b = float(model.intercept_)
    return np.concatenate([w, [b]])

def gp_adaptiveinject(gp):
    """Adaptive Orthogonal Symbolic Injection Module"""
    
    adaptgen = gp.config['runcontrol']['adaptgen']
    
    if gp.config['runcontrol']['adaptinject'] and not gp.state['adaptinjected'] and gp.state['stallgen'] == adaptgen:
        
        # 1. Get objective values and prediction errors
        num_z = len(gp.state['injected_expression'])
        obj = gp.individuals['objective']['ensemble']
        minimise = gp.config['runcontrol']['minimisation']
        yp_tr = gp.individuals['yp']['ensemble']['train']
        yp_val = gp.individuals['yp']['ensemble']['validation']
        yp_ts = gp.individuals['yp']['ensemble']['test']
        ytr = gp.userdata['ytrain']
        yval = gp.userdata['yval']
        yts = gp.userdata['ytest']
        num_pop = gp.config['runcontrol']['num_pop']
        pop_size = gp.config['runcontrol']['pop_size']
        w_genes = gp.individuals['weight_genes']
        w_ensemble = gp.individuals['ensemble_weight']
        random_state = gp.config['runcontrol']['random_state']
        
        if yval is not None:
            y_combined = np.concatenate([ytr, yval])
        else:
            y_combined = ytr
        
        pop = gp.population
        
        z1_token = f'z{num_z + 1}'
        z2_token = f'z{num_z + 1}'
        
        # 2. Identify z1 (best individual)
        idx_z1 = np.argmin(obj) if minimise else np.argmax(obj)
        
        if yval is not None:
            z1_pred = np.concatenate([yp_tr[idx_z1], yp_val[idx_z1]])
        else:
            z1_pred = yp_tr[idx_z1]
        
        # 3. Identify top 10% individuals
        top_k = max(2, pop_size // 25)
        top_idxs = np.argsort(obj)[:top_k] if minimise else np.argsort(obj)[-top_k:]
        
        # 4. Find z2: least correlated error with z1
        z1_err = z1_pred - y_combined
        best_corr = np.inf
        idx_z2 = None
        for i in top_idxs:
            if i == idx_z1:
                continue
            
            if yval is not None:
                err = np.concatenate([yp_tr[i], yp_val[i]]) - y_combined
            else:
                err = yp_tr[i] - y_combined
            
            corr = abs(np.corrcoef(z1_err, err)[0,1])
            if corr < best_corr:
                best_corr = corr
                idx_z2 = i
        
        if yval is not None:
            z2_pred = np.concatenate([yp_tr[idx_z2], yp_val[idx_z2]])
        else:
            z2_pred = yp_tr[idx_z2]
        
        # 5. Construct 4-gene candidate: [z1, z2, z1+z2, z1*z2]
        g1, g2 = z1_pred, z2_pred
        X_full = np.column_stack([g1, g2, g1 + g2, g1 * g2])
        gene_names = [f'{z1_token}', f'{z2_token}', f'plus({z1_token},{z2_token})', f'times({z1_token},{z2_token})']
        
        # 6. Evaluate performance
        X_combined = np.concatenate([g1.reshape(-1,1), g2.reshape(-1,1)], axis=0)
        mse_z1 = cross_val_mse(g1.reshape(-1, 1), y_combined, random_state)
        mse_injected = cross_val_mse(X_full, y_combined, random_state)
        
        # 7. Prune if needed
        epsilon = 1e-4
        if mse_injected > mse_z1 + epsilon:
            X_final, kept_genes = adaptive_gene_pruning(X_full, gene_names, y_combined, mse_z1, epsilon = epsilon, random_state=random_state)
        else:
            X_final = X_full
            kept_genes = gene_names

        z1_z2_present = detect_z1_z2_presence(kept_genes, z1_token, z2_token)
        selected_idxs = []
        if z1_z2_present[0]:
            selected_idxs.append(idx_z1)
        if z1_z2_present[1]:
            selected_idxs.append(idx_z2)
        
        # Stack only the selected predictions
        if len(gp.state['ztrain']) == 0:
            gp.state['ztrain'] = np.column_stack([yp_tr[i] for i in selected_idxs])
        else:
            gp.state['ztrain'] = np.column_stack([gp.state['ztrain'], *[yp_tr[i] for i in selected_idxs]])

        
        if yval is not None:
            if len(gp.state['zval']) == 0:
                gp.state['zval'] = np.column_stack([yp_val[i] for i in selected_idxs])
            else:
                gp.state['zval'] = np.column_stack([gp.state['zval'], *[yp_val[i] for i in selected_idxs]])
         
        if yts is not None:
            if len(gp.state['ztest']) == 0:
                gp.state['ztest'] = np.column_stack([yp_ts[i] for i in selected_idxs])
            else:
                gp.state['ztest'] = np.column_stack([gp.state['ztest'], *[yp_ts[i] for i in selected_idxs]])
        
        # building the expression to save
        w_g1 = []
        w_g2 = []
        indiv1 = []
        indiv2 = []
        for id_pop in range(num_pop):
            w_g1.append(w_genes[id_pop][idx_z1])
            w_g2.append(w_genes[id_pop][idx_z2])
            for gene1 in pop[id_pop][idx_z1]:
                indiv1.append(copy.deepcopy(gene1))
            for gene2 in pop[id_pop][idx_z2]:
                indiv2.append(copy.deepcopy(gene2))
        
        w_e1 = w_ensemble[idx_z1]
        w_e2 = w_ensemble[idx_z2]
        
        w_n1 = calculate_weights(w_g1, w_e1)
        w_n2 = calculate_weights(w_g2, w_e2)
        
        exp1 = build_expression(w_n1, indiv1)
        exp2 = build_expression(w_n2, indiv2)
        
        # Replace z_i in exp1 and exp2 with prior injected expressions, if any
        if gp.state['injected_expression']:
            exp_n1, exp_n2 = exp1, exp2
            for i, replacement in enumerate(gp.state['injected_expression']):
                z_token = f'z{i + 1}'
                exp_n1 = exp_n1.replace(z_token, replacement)
                exp_n2 = exp_n2.replace(z_token, replacement)
        else:
            exp_n1, exp_n2 = exp1, exp2

        
        # Store the final (possibly substituted) expressions
        if z1_z2_present[0]:
            gp.state['injected_expression'].append(exp_n1)
        if z1_z2_present[1]:
            gp.state['injected_expression'].append(exp_n2)
        
            
        for id_pop in range(num_pop):
            pop[id_pop][idx_z2]  = copy.deepcopy(kept_genes)
            new_w = retrain_weights(X_final, y_combined, random_state)
            gp.individuals["weight_genes"][id_pop][idx_z2] = new_w
            
        # 9. Mark state as injected
        gp.state['adaptinjected'] = True
        gp.population = copy.deepcopy(pop)
        gp.state['num_adaptinject'] += 1
        print("Adaptive Orthogonal Symbolic Injection is being performed")
        
        
        
        
        
        