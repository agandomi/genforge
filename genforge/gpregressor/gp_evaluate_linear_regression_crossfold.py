# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
import warnings
from .StandardScaler import StandardScaler


def gp_evaluate_linear_regression_crossfold(args):
    """
    Train a Lasso regression model with internal cross-validation and evaluate
    predictions and losses on train, validation, and test sets.
    """
    
    ytr, yval, yts, \
        l1_ratio, alphas, n_alphas, \
        eps, fit_intercept, copy_x, max_iter, \
        tol, cv, n_jobs, verbose, positive, selection, random_seed, id_pop, id_ind, \
        gene_out_tr, gene_out_val, gene_out_ts = args

    scaler = StandardScaler()    
    # Combine training and validation sets
    if yval is not None:
        X_combined = np.vstack([gene_out_tr, gene_out_val])
        y_combined = np.concatenate([ytr, yval])
    else:
        X_combined = gene_out_tr
        y_combined = ytr
    
    scaler.fit(X_combined)
    
    # print([np.max(gene_out_tr), np.max(gene_out_val), np.max(gene_out_ts)])
    X_tr_std = scaler.transform(gene_out_tr)
    X_val_std = scaler.transform(gene_out_val) if gene_out_val is not None else None
    X_ts_std = scaler.transform(gene_out_ts) if gene_out_ts is not None else None
    
    if yval is not None:
        X_combined_std = np.vstack([X_tr_std, X_val_std])
    else:
        X_combined_std = X_tr_std
    # Fit LassoCV (with internal k-fold CV to find best alpha)
    # [0.5, 0.7, 0.9, 0.95, 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas, n_alphas=n_alphas, eps=eps, fit_intercept=fit_intercept,\
                              copy_X=copy_x, max_iter=max_iter, tol=tol, cv=cv, n_jobs=n_jobs, verbose=verbose,\
                                  positive=positive, selection=selection, random_state=random_seed).fit(X_combined_std, y_combined)
    

    # Predict
    yp_tr = model.predict(X_tr_std)
    yp_val = model.predict(X_val_std) if X_val_std is not None else None
    yp_ts = model.predict(X_ts_std) if X_ts_std is not None else None

    # Compute losses
    loss_tr = mean_squared_error(ytr, yp_tr)
    loss_val = mean_squared_error(yval, yp_val) if gene_out_val is not None else None
    loss_ts = mean_squared_error(yts, yp_ts) if gene_out_ts is not None else None
    
    w_original = model.coef_ / scaler.scale_
    b_original = model.intercept_ - np.sum((model.coef_ * scaler.mean_) / scaler.scale_)

    # Combine weights and bias
    weights = w_original.reshape(-1, 1)         # Shape: (n_genes, 1)
    bias = np.array([[b_original]])        # Shape: (1, 1)
    weights_with_biases = np.vstack([weights, bias])  # Final shape: (n_genes + 1, 1)

    # Package results
    results = [yp_tr, yp_val, yp_ts, loss_tr, loss_val, loss_ts, weights_with_biases]
    return results
