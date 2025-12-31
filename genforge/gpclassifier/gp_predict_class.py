# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025
import numpy as np
from .gp_evaluate_tree_class import gp_evaluate_tree
from .SoftmaxModel import SoftmaxModel

def _classwise_weighted_sum(prob_list, weights):
    """
    prob_list: list of [N, C] (one per population)
    weights:   list (len=P) of lists (len=C) -> weights[pop][class]
    Returns [N, C] classwise weighted average, same as training.
    """
    N, C = prob_list[0].shape
    P = len(prob_list)
    out = np.zeros((N, C), dtype=float)
    for cls in range(C):
        # stack the P population probs for this class: [N, P]
        pc = np.stack([prob_list[p][:, cls] for p in range(P)], axis=1)
        wc = np.asarray([weights[p][cls] for p in range(P)], dtype=float)  # per-class normalized weights
        s = wc.sum()
        if s <= 0:
            # fallback to uniform if somehow degenerate
            wc = np.ones_like(wc) / P
        else:
            wc = wc / s
        out[:, cls] = pc @ wc
    return out

def _eval_individual_probs(gp, X, id_pop, id_ind):
    """
    Replicate the isolated-probability path used during training.
    - Evaluate each gene -> tanh(...)
    - Use SoftmaxModel.predict() with saved weights/bias
    """
    Xp = X

    # Evaluate all genes of the chosen individual -> feature matrix [N, G]
    individual = gp.population[id_pop][id_ind]
    function_map = gp.config['nodes']['functions']['function'][id_pop]
    N = Xp.shape[0]
    G = len(individual)
    features = np.empty((N, G), dtype=float)
    for g, gene in enumerate(individual):
        features[:, g] = np.tanh(gp_evaluate_tree(gene, Xp, function_map))

    # Restore weights & bias exactly as stored during training
    wb = gp.individuals['weight_genes'][id_pop][id_ind]
    # training stores W|b as a single [C, G+1] matrix: last column is bias
    W = wb[:, :G]
    b = wb[:, G:G+1]     # shape (C,)

    # Build a SoftmaxModel compatible with the saved shape and let it do predict()
    # This ensures same math as in evolve().
    model = SoftmaxModel(
        xtrain=features,             # used for shape inside the model
        ytrain=None,
        xval=None,
        yval=None,
        xts=None,
        yts=None,
        num_class=gp.userdata['num_class'],
        initializer=gp.config['softmax']['initializer'][id_pop],
        random_seed=gp.config['runcontrol']['random_state'],
        batch_size=gp.config['softmax']['batch_size'][id_pop],
        epochs=gp.config['softmax']['epochs'][id_pop],
        patience=gp.config['softmax']['patience'][id_pop],
        shuffle=gp.config['softmax']['shuffle'][id_pop],
        buffer_size=gp.config['softmax']['buffer_size'][id_pop],
        verbose=gp.config['softmax']['verbose'][id_pop],
        regularization=gp.config['softmax']['regularization'][id_pop],
        regularization_rate=gp.config['softmax']['regularization_rate'][id_pop],
        optimizer_type=gp.config['softmax']['optimizer_type'][id_pop],
        optimizer_param=gp.config['softmax']['optimizer_param'][id_pop],
    )
    # overwrite learned params
    model.weight = W.copy()
    model.bias   = b.copy()

    # IMPORTANT: use the model's predict() to match training-time probability math
    prob = model.predict(features)   # [N, C]
    yp = np.argmax(prob, axis=1)
    return prob, yp

def _eval_population_all(gp, X, id_pop):
    """
    Evaluate all individuals in a population on X.
    Returns:
        probs_list: list of length pop_size, each [N, C]
        y_list:     list of length pop_size, each [N]
    """
    pop_size = len(gp.population[id_pop])
    probs_list, y_list = [], []
    for id_ind in range(pop_size):
        prob, yp = _eval_individual_probs(gp, X, id_pop, id_ind)
        probs_list.append(prob)
        y_list.append(yp)
    return probs_list, y_list

def gp_predict(gp, X, ensemble_row: int = 0, return_proba: bool = False,
               *, mode: str = "ensemble", id_pop: int | None = None, id_ind: int | None = None):
    """
    Predict on new data X with flexible modes.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to predict on.
    ensemble_row : int, default=0
        Which ensemble row to use (0-based) when mode="ensemble".
    return_proba : bool, default=False
        If True, return class probabilities [N, C]; otherwise return class indices [N].
    mode : {"ensemble","isolated","population","all"}, default="ensemble"
        - "ensemble": combine chosen individuals per population using classwise weights.
        - "isolated": predict using a single individual (requires id_pop and id_ind).
        - "population": predict using ALL individuals of a given population (requires id_pop).
        - "all": predict using ALL individuals of ALL populations.
    id_pop : int, optional
        Population index (required for "isolated" and "population").
    id_ind : int, optional
        Individual index within population (required for "isolated").

    Returns
    -------
    np.ndarray or list
        - mode="ensemble": [N, C] if return_proba else [N]
        - mode="isolated": [N, C] if return_proba else [N]
        - mode="population": list of length pop_size; each item is [N, C] or [N]
        - mode="all": list of length num_pop; each is a list (len=pop_size) of [N, C] or [N]
    """
    X = np.asarray(X)
    if X.ndim != 2:
        gp.error("X must be a 2D array of shape (n_samples, n_features).", ValueError)

    num_pop = gp.config['runcontrol']['num_pop']
    num_class = gp.userdata['num_class']

    if mode == "ensemble":
        # Validate ensemble availability
        ens_total = len(gp.individuals.get('ensemble_idx', []))
        if ens_total == 0:
            gp.error("No ensembles available. Run evolve() first.", RuntimeError)
        if not (0 <= ensemble_row < ens_total):
            gp.error(f"ensemble_row out of range: got {ensemble_row}, valid 0..{ens_total-1}.", IndexError)

        # Which isolated individual per population was chosen for this ensemble?
        chosen = gp.individuals['ensemble_idx'][ensemble_row]      # shape (num_pop,)
        weights = gp.individuals['ensemble_weight'][ensemble_row]  # list-of-lists [P][C] (classwise normalized)

        per_pop_probs = []
        for p in range(num_pop):
            ind = chosen[p]
            prob, _ = _eval_individual_probs(gp, X, p, ind)
            if prob.shape[1] != num_class:
                gp.error(f"Population {p} predicted {prob.shape[1]} classes, expected {num_class}.")
            per_pop_probs.append(prob)

        # Combine per-population probabilities using classwise normalized weights
        prob_ens = _classwise_weighted_sum(per_pop_probs, weights)
        return prob_ens if return_proba else np.argmax(prob_ens, axis=1)

    elif mode == "isolated":
        # Single individual prediction
        if id_pop is None or id_ind is None:
            gp.error("mode='isolated' requires both id_pop and id_ind.", ValueError)
        if not (0 <= id_pop < num_pop):
            gp.error(f"id_pop out of range: got {id_pop}, valid 0..{num_pop-1}.", IndexError)
        if not (0 <= id_ind < len(gp.population[id_pop])):
            gp.error(f"id_ind out of range for population {id_pop}: got {id_ind}, "
                     f"valid 0..{len(gp.population[id_pop]) - 1}.", IndexError)

        prob, yp = _eval_individual_probs(gp, X, id_pop, id_ind)
        return prob if return_proba else yp

    elif mode == "population":
        # All individuals in a chosen population
        if id_pop is None:
            gp.error("mode='population' requires id_pop.", ValueError)
        if not (0 <= id_pop < num_pop):
            gp.error(f"id_pop out of range: got {id_pop}, valid 0..{num_pop-1}.", IndexError)

        probs_list, y_list = _eval_population_all(gp, X, id_pop)
        return probs_list if return_proba else y_list

    elif mode == "all":
        # All individuals of all populations
        out = []
        for p in range(num_pop):
            probs_list, y_list = _eval_population_all(gp, X, p)
            out.append(probs_list if return_proba else y_list)
        return out

    else:
        gp.error(f"Unknown mode '{mode}'. Use 'ensemble', 'isolated', 'population', or 'all'.", ValueError)
