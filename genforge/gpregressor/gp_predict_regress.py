# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
from .gp_evaluate_tree_regress import gp_evaluate_tree

def _extract_wb_vector(gp, wb, expected_g):
    """
    Normalize various shapes for a vector of length expected_g+1 whose last
    element is the intercept (bias). Returns (w, b) with w shape [G], b float.
    """
    arr = np.asarray(wb)
    if arr.ndim == 1:
        if arr.size != expected_g + 1:
            gp.error(f"Saved weight length {arr.size} != G+1 ({expected_g+1}).", ValueError)
        return arr[:expected_g].astype(float), float(arr[-1])
    if arr.ndim == 2:
        r, c = arr.shape
        if (r, c) == (expected_g + 1, 1):
            return arr[:expected_g, 0].astype(float), float(arr[expected_g, 0])
        if (r, c) == (1, expected_g + 1):
            return arr[0, :expected_g].astype(float), float(arr[0, expected_g])
    gp.error(f"Unsupported weight matrix shape {arr.shape} for G={expected_g}.", ValueError)


def _predict_individual(gp, X, id_pop: int, id_ind: int):
    """
    Reproduce training-time isolated prediction for a SINGLE individual:
      - evaluate each gene on X
      - apply saved ElasticNetCV weights (already mapped back to original scale)
    Returns: y_pred [N]
    """
    # Build feature matrix [N, G] by evaluating each gene (no extra tanh/clip).
    indiv = gp.population[id_pop][id_ind]
    func_map = gp.config["nodes"]["functions"]["function"][id_pop]
    N = int(X.shape[0])
    G = len(indiv)
    feats = np.empty((N, G), dtype=float)

    # If your gene operators can reference 'z' inputs, pass z=None here.
    # Training-time ztrain/zval/… are for adaptive-injection; for fresh X we use None.
    z = None
    for g, gene in enumerate(indiv):
        vals, _pen = gp_evaluate_tree(gene, X, z, func_map)
        feats[:, g] = vals

    # Pull saved weights | bias (already de-standardized at training time)
    wb_saved = gp.individuals["weight_genes"][id_pop][id_ind]
    w, b = _extract_wb_vector(gp, wb_saved, G)

    # Linear prediction
    y = feats @ w + b
    return y


def _extract_ensemble_wb(gp, wb, expected_p):
    """
    Ensemble weight vector length expected_p+1 (per-pop weights + bias).
    Returns (w_pop, b) with w_pop shape [P], b float.
    """
    arr = np.asarray(wb)
    if arr.ndim == 1:
        if arr.size != expected_p + 1:
            gp.error(f"Ensemble weight length {arr.size} != P+1 ({expected_p+1}).", ValueError)
        return arr[:expected_p].astype(float), float(arr[-1])
    if arr.ndim == 2:
        r, c = arr.shape
        if (r, c) == (expected_p + 1, 1):
            return arr[:expected_p, 0].astype(float), float(arr[expected_p, 0])
        if (r, c) == (1, expected_p + 1):
            return arr[0, :expected_p].astype(float), float(arr[0, expected_p])
    gp.error(f"Unsupported ensemble weight shape {arr.shape} for P={expected_p}.", ValueError)


def gp_predict(
    gp,
    X,
    *,
    ensemble_row: int = 0,
    mode: str = "ensemble",
    id_pop: int | None = None,
    id_ind: int | None = None,
):
    """
    Predict for regression runs.

    Parameters
    ----------
    gp : gpregressor instance
    X  : array-like of shape (n_samples, n_features)
         Features must be in the same order/scale as at training time.
    mode : {"ensemble","isolated"}
        - "ensemble" (default): use the stored ensemble (multi-pop only).
        - "isolated": predict from a specific (id_pop, id_ind).
    ensemble_row : int
        mode saved ensemble row to use (0-based). Ignored for isolated mode.
    id_pop, id_ind : int
        Required when mode="isolated": choose a population and an individual.

    Returns
    -------
    y_pred : ndarray, shape (n_samples,)
    """
    X = np.asarray(X)
    if X.ndim != 2:
        gp.error("X must be a 2D array of shape (n_samples, n_features).", ValueError)

    P = int(gp.config["runcontrol"]["num_pop"])
    mode = (mode or "ensemble").strip().lower()

    # --------- isolated path ----------
    if mode == "isolated":
        if id_pop is None or id_ind is None:
            gp.error("mode='isolated' requires both id_pop and id_ind.", ValueError)
        if not (0 <= int(id_pop) < P):
            gp.error(f"id_pop out of range [0..{P-1}].", IndexError)
        if not (0 <= int(id_ind) < len(gp.population[int(id_pop)])):
            gp.error(f"id_ind out of range for population {id_pop}.", IndexError)
        return _predict_individual(gp, X, int(id_pop), int(id_ind))

    # --------- ensemble path ----------
    if mode != "ensemble":
        gp.warning(f"Unknown mode={mode!r}; falling back to 'ensemble'.")

    if P == 1:
        # Single-pop runs don't build ensembles; fallback to best isolated (or first).
        # Use rank from training if available; otherwise take individual 0.
        try:
            # If you rank by validation fitness:
            order = gp.track["rank"]["fitness"]["isolated"]["validation"][:, 0]
            best_id = int(order[0])
        except Exception:
            best_id = 0
        return _predict_individual(gp, X, 0, best_id)

    # Multi-pop ensembles expected
    ens_idx_all = gp.individuals.get("ensemble_idx", None)
    ens_w_all   = gp.individuals.get("ensemble_weight", None)
    if ens_idx_all is None or ens_w_all is None:
        gp.error("No ensemble data present; run evolve() first.", RuntimeError)

    if not (0 <= int(ensemble_row) < len(ens_idx_all)):
        gp.error(f"ensemble_row out of range: got {ensemble_row}, valid 0..{len(ens_idx_all)-1}.", IndexError)

    chosen_inds = np.asarray(ens_idx_all[int(ensemble_row)]).astype(int)  # length P
    if chosen_inds.size != P:
        gp.error(f"Stored ensemble_idx length {chosen_inds.size} != num_pop {P}.", ValueError)

    # Per-pop isolated predictions on X
    preds = []
    for p in range(P):
        yp = _predict_individual(gp, X, p, int(chosen_inds[p]))
        preds.append(yp)
    M = np.column_stack(preds)  # [N, P]

    # Stored ensemble weights (normalized at training) + bias
    w_pop, b = _extract_ensemble_wb(gp, ens_w_all[int(ensemble_row)], P)  # w_pop [P], b scalar

    # Final ensemble prediction
    y_ens = M @ w_pop + b
    return y_ens
