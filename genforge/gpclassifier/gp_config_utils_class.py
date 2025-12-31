# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
from __future__ import annotations
from typing import Any, List, Tuple, Dict
import os, importlib, inspect
import numpy as np

from .gp_config_data_class import (
    ClassifierConfig, Diagnostics,
)

# ------------------------- RNG initialization -------------------------

def _init_rng(gp, random_state):
    """
    Normalize the user-provided random_state into a np.random.Generator.
    Preserves the original random_state in gp.config['runcontrol']['random_state'].
    """
    import numpy as np

    if random_state is None:
        return np.random.default_rng()

    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(int(random_state))

    if isinstance(random_state, np.random.Generator):
        return random_state

    if isinstance(random_state, np.random.RandomState):
        gp.warning(
            "Using legacy np.random.RandomState; consider switching to np.random.default_rng(int)."
        )
        return np.random.Generator(np.random.MT19937(random_state.randint(0, 2**32 - 1)))

    gp.error(
        f"'runcontrol.random_state' must be None, int, np.random.Generator, or np.random.RandomState; "
        f"got {type(random_state).__name__}={random_state!r}",
        ValueError,
    )

def _rng_to_seed(gp, user_random_state, rng):
    """
    Derive a reproducible seed for Softmax/ElasticNet.

    Parameters
    ----------
    user_random_state : None | int | np.random.Generator | np.random.RandomState
        The original input from cfg.runcontrol.random_state (preserved).
    rng : np.random.Generator
        The normalized Generator created by _init_rng.

    Returns
    -------
    seed : None | int
        None if user gave None, the same integer if user gave int,
        or a derived int seed if user gave a Generator/RandomState.
    """
    import numpy as np

    if user_random_state is None:
        return None
    if isinstance(user_random_state, (int, np.integer)):
        return int(user_random_state)
    if isinstance(user_random_state, (np.random.Generator, np.random.RandomState)):
        return int(rng.integers(0, 2**32 - 1))
    # fallback (shouldn’t normally happen because _init_rng already validated)
    gp.error(
        f"'runcontrol.random_state' must be None, int, np.random.Generator, or np.random.RandomState; "
        f"got {type(user_random_state).__name__}={user_random_state!r}",
        ValueError,
    )

# ------------------------- tiny predicates -------------------------

def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)

def _is_float(x) -> bool:
    return isinstance(x, (float, int, np.floating, np.integer)) and not isinstance(x, bool)

def _is_bool(x) -> bool:
    return isinstance(x, (bool, np.bool_))

def _is_range_pair(e) -> bool:
    return isinstance(e, (list, tuple)) and len(e) == 2 and _is_float(e[0]) and _is_float(e[1])

# ------------------------- basic ensure helpers -------------------------

def _ensure_int(gp, name, v):
    if not _is_int(v):
        gp.error(f"'{name}' must be an integer; got {type(v).__name__}={v!r}", ValueError)
    return int(v)

def _ensure_float(gp, name, v):
    if not _is_float(v):
        gp.error(f"'{name}' must be a float; got {type(v).__name__}={v!r}", ValueError)
    return float(v)

def _ensure_bool(gp, name, v):
    if not _is_bool(v):
        gp.error(f"'{name}' must be a bool; got {type(v).__name__}={v!r}", ValueError)
    return bool(v)

def _ensure_prob01(gp, name, v, *, inclusive: bool = True):
    vf = _ensure_float(gp, name, v)
    ok = (0.0 <= vf <= 1.0) if inclusive else (0.0 < vf < 1.0)
    if not ok:
        gp.error(f"'{name}' must be within [0,1]{' (exclusive)' if not inclusive else ''}; got {vf}", ValueError)
    return vf

def _validate_range_pair(gp, name, e):
    if not _is_range_pair(e):
        gp.error(f"'{name}' entries must be [min, max].", ValueError)
    lo, hi = float(e[0]), float(e[1])
    if lo >= hi:
        gp.error(f"'{name}' min must be < max; got {lo} >= {hi}", ValueError)
    return [lo, hi]

def _as_list_n(gp, name, v, n, *, validator=None, allow_scalar=True):
    """Broadcast scalar → len n, or validate list length. Optionally per-element validator."""
    if isinstance(v, (list, tuple, np.ndarray)):
        lst = list(v)
        if len(lst) == 1 and n > 1:
            lst = lst * n
        elif len(lst) != n:
            gp.error(f"'{name}' must have length {n}; got length {len(lst)}", ValueError)
    else:
        if not allow_scalar:
            gp.error(f"'{name}' must be a list of length {n}; got {v!r}", ValueError)
        lst = [v] * n
    if validator:
        lst = [validator(gp, name, e) for e in lst]
    return lst

# ------------------------- arrays & labels -------------------------

def _flatten_labels(gp, arr, name):
    if arr is None:
        return None
    if not isinstance(arr, np.ndarray):
        gp.error(f"'{name}' must be a numpy array; got {type(arr).__name__}", ValueError)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.ravel()
    if arr.ndim == 1:
        return arr
    gp.error(f"'{name}' must be 1D or 2D with a single column; got shape {arr.shape}", ValueError)

def _flatten_2d(gp, arr, name):
    if not isinstance(arr, np.ndarray):
        gp.error(f"'{name}' must be a numpy array; got {type(arr).__name__}", ValueError)
    if arr.ndim != 2:
        gp.error(f"'{name}' must be 2D (n_samples, n_features); got shape {arr.shape}", ValueError)
    return arr

def _unique_labels(*arrays):
    cats = []
    for a in arrays:
        if a is not None:
            cats.append(np.asarray(a).ravel())
    if not cats:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(cats))

def _binary_mapping(y, num_classes):
    y = np.asarray(y, dtype=int)
    out = np.zeros((len(y), num_classes), dtype=int)
    out[np.arange(len(y)), y] = 1
    return out

# ------------------------- filesystem / plotting -------------------------

def _ensure_dir(gp, name: str, path, *, default_to_cwd: bool = False) -> str:
    """Accept None or str path. If None (or "" and default_to_cwd), use CWD. Ensure directory exists."""
    if path is None or (default_to_cwd and isinstance(path, str) and path.strip() == ""):
        return os.getcwd()
    if not isinstance(path, str):
        gp.error(f"'{name}' must be None or a string path.", ValueError)
    expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    try:
        os.makedirs(expanded, exist_ok=True)
    except Exception as e:
        gp.error(f"Could not create directory for '{name}' at {expanded!r}: {e}", ValueError)
    return expanded

def _validate_plot_formats(gp, fmt):
    allowed = {
        "png", "jpg", "jpeg", "tif", "tiff", "pdf", "svg", "svgz",
        "ps", "eps", "raw", "rgba", "pgf", "emf"
    }
    if isinstance(fmt, str):
        lst = [fmt.lower()]
    elif isinstance(fmt, (list, tuple)):
        lst = [str(x).lower() for x in fmt]
    else:
        gp.error("'runcontrol.plotformat' must be a string or list of strings.", ValueError)
    bad = [x for x in lst if x not in allowed]
    if bad:
        gp.error(f"'runcontrol.plotformat' contains unsupported formats {bad}. Allowed: {sorted(allowed)}", ValueError)
    return lst

def _validate_backend(gp, be):
    allowed = {"auto", "QtAgg", "TkAgg", "MacOSX", "Agg"}
    if not isinstance(be, str):
        gp.error("'runcontrol.plotbackend' must be a string.", ValueError)
    if be not in allowed:
        gp.error(f"'runcontrol.plotbackend' must be one of {sorted(allowed)}; got {be!r}", ValueError)
    return be

# ------------------------- population / indices -------------------------

def _infer_num_pop(gp, xtrain, pop_idx_param, num_pop_param):
    n_features = xtrain.shape[1]
    if pop_idx_param is None:
        pop_idx = None
    else:
        if isinstance(pop_idx_param, (list, tuple)):
            if all(_is_int(x) for x in pop_idx_param):
                pop_idx = [list(pop_idx_param)]
            elif all(isinstance(x, (list, tuple)) for x in pop_idx_param):
                pop_idx = [list(map(int, grp)) for grp in pop_idx_param]
            else:
                gp.error("'user.pop_idx' must be a list of ints or list of lists of ints.", ValueError)
        else:
            gp.error("'user.pop_idx' must be a list (or list of lists).", ValueError)
        # bounds check
        for i, grp in enumerate(pop_idx):
            for j in grp:
                if not _is_int(j) or not (0 <= int(j) < n_features):
                    gp.error(f"'user.pop_idx' has invalid column {j} for population {i} (valid 0..{n_features-1}).",
                             ValueError)

    if num_pop_param is None:
        num_pop = 1 if pop_idx is None else len(pop_idx)
        if pop_idx is None:
            pop_idx = [list(range(n_features))]
    else:
        if not _is_int(num_pop_param) or int(num_pop_param) < 1:
            gp.error("'runcontrol.num_pop' must be an integer >= 1 or None.", ValueError)
        desired = int(num_pop_param)
        if pop_idx is None:
            if desired == 1:
                pop_idx = [list(range(n_features))]
            else:
                gp.error("For multi-population GP, 'user.pop_idx' must be provided.", ValueError)
        else:
            if desired != len(pop_idx):
                gp.error("'runcontrol.num_pop' does not match len(user.pop_idx).", ValueError)
        num_pop = desired

    return int(num_pop), pop_idx

# ------------------------- functions import -------------------------

_OPTIMIZER_DEFAULTS = {
    "sgd":     {"learning_rate": 0.01, "momentum": 0.0, "nesterov": False, "weight_decay": 0.0,
                "dampening": 0.0, "decay": 0.0, "clipnorm": None, "clipvalue": None},
    "adam":    {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-7,
                "amsgrad": False, "weight_decay": 0.0, "clipnorm": None, "clipvalue": None},
    "rmsprop": {"learning_rate": 0.001, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-7,
                "centered": False, "weight_decay": 0.0, "clipnorm": None, "clipvalue": None},
    "adamw":   {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
                "weight_decay": 0.01, "clipnorm": None, "clipvalue": None},
    "nadam":   {"learning_rate": 0.002, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-7,
                "weight_decay": 0.0, "clipnorm": None, "clipvalue": None},
    "adagrad": {"learning_rate": 0.01, "epsilon": 1e-7, "initial_accumulator_value": 0.1,
                "weight_decay": 0.0, "clipnorm": None, "clipvalue": None},
    "adadelta":{"learning_rate": 1.0, "rho": 0.95, "epsilon": 1e-7,
                "weight_decay": 0.0, "clipnorm": None, "clipvalue": None},
    "ftrl":    {"learning_rate": 0.001, "learning_rate_power": -0.5, "initial_accumulator_value": 0.1,
                "l1_regularization_strength": 0.0, "l2_regularization_strength": 0.0,
                "weight_decay": 0.0, "clipnorm": None, "clipvalue": None},
    "adamax":  {"learning_rate": 0.002, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-7,
                "weight_decay": 0.0, "clipnorm": None, "clipvalue": None},
    "sgdnm":   {"learning_rate": 0.01, "momentum": 0.9, "nesterov": True,
                "weight_decay": 0.0, "dampening": 0.0, "decay": 0.0,
                "clipnorm": None, "clipvalue": None},
    "lbfgs":   {"learning_rate": 1.0, "max_iter": 20, "max_eval": None,
                "tolerance_grad": 1e-5, "tolerance_change": 1e-9,
                "history_size": 100, "line_search_fn": "strong_wolfe"},
}
def _normalize_opt_name(n: str) -> str:
    n = str(n).lower()
    return "lbfgs" if n == "lbgfs" else n

def _import_operator_function(gp, func_name: str):
    module_name = func_name.lower()
    last_err = None
    for mod in (module_name, f"genforge.{module_name}"):
        try:
            module = importlib.import_module(mod)
            fn = getattr(module, func_name)
            sig = inspect.signature(fn)
            nargs = sum(1 for p in sig.parameters.values()
                        if p.default == inspect.Parameter.empty and
                           p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))
            return fn, int(nargs)
        except Exception as e:
            last_err = e
    gp.error(f"Could not import function '{func_name}': {last_err}", ImportError)

# ------------------------- Softmax helpers -------------------------

def _broadcast_optimizer_types(gp, v, n):
    if isinstance(v, str):
        types = [v] * n
    elif isinstance(v, (list, tuple)):
        if len(v) == 1 and n > 1:
            types = list(v) * n
        elif len(v) != n:
            gp.error("'softmax.optimizer_type' length must equal num_pop.", ValueError)
        else:
            types = list(v)
    else:
        gp.error("'softmax.optimizer_type' must be a string or list of strings.", ValueError)
    types = [_normalize_opt_name(t) for t in types]
    for t in types:
        if t not in _OPTIMIZER_DEFAULTS:
            gp.error(f"Unknown optimizer type '{t}'. Allowed: {sorted(_OPTIMIZER_DEFAULTS)}", ValueError)
    return types

def _broadcast_optimizer_params(gp, opt_types, params, n, diags: Diagnostics):
    if params is None:
        plist = [None] * n
    elif isinstance(params, dict):
        plist = [params] * n
    elif isinstance(params, (list, tuple)):
        if len(params) == 1 and n > 1:
            plist = list(params) * n
        elif len(params) != n:
            gp.error("'softmax.optimizer_param' length must equal num_pop.", ValueError)
        else:
            plist = list(params)
    else:
        gp.error("'softmax.optimizer_param' must be None, a dict, or a list of dicts.", ValueError)

    out = []
    for i, (t, pd) in enumerate(zip(opt_types, plist)):
        defaults = _OPTIMIZER_DEFAULTS[t].copy()
        if pd is None:
            out.append(defaults)
            continue
        if not isinstance(pd, dict):
            gp.error(f"'softmax.optimizer_param[{i}]' must be a dict; got {type(pd).__name__}", ValueError)
        unknown = sorted(set(pd) - set(defaults))
        if unknown:
            diags.warn(f"softmax.optimizer_param[{i}] for '{t}' has unknown keys {unknown}; they will be ignored.")
        merged = defaults.copy()
        for k, v in pd.items():
            if k in defaults:
                merged[k] = v
        out.append(merged)
    return out

# ------------------------- Resolve classifier config -------------------------

def resolve_classifier_config(gp, cfg: ClassifierConfig, *, return_diagnostics: bool = False):
    diags = Diagnostics()

    # ---- Data checks
    xtr = _flatten_2d(gp, cfg.user.xtrain, "user.xtrain")
    ytr = _flatten_labels(gp, cfg.user.ytrain, "user.ytrain")
    if len(xtr) != len(ytr):
        gp.error("Training X and y must have the same number of rows.", ValueError)

    if cfg.user.xval is None:
        xval = None; yval = None
    else:
        xval = _flatten_2d(gp, cfg.user.xval, "user.xval")
        yval = _flatten_labels(gp, cfg.user.yval, "user.yval")
        if xval.shape[1] != xtr.shape[1]:
            gp.error("Validation X must have the same number of columns as training X.", ValueError)
        if len(xval) != len(yval):
            gp.error("Validation X and y must have the same number of rows.", ValueError)

    if cfg.user.xtest is None:
        xts = None; yts = None
    else:
        xts = _flatten_2d(gp, cfg.user.xtest, "user.xtest")
        yts = _flatten_labels(gp, cfg.user.ytest, "user.ytest")
        if xts.shape[1] != xtr.shape[1]:
            gp.error("Test X must have the same number of columns as training X.", ValueError)
        if len(xts) != len(yts):
            gp.error("Test X and y must have the same number of rows.", ValueError)

    # ---- labels normalization to 0..K-1
    all_labels = _unique_labels(ytr, yval, yts)
    if all_labels.size == 0:
        gp.error("No labels found in y arrays.", ValueError)
    expected = np.arange(all_labels.size, dtype=int)
    if not np.array_equal(all_labels, expected):
        gp.warning(f"Class labels {all_labels.tolist()} remapped to {expected.tolist()} (0..K-1).")
        mapping = {lab: i for i, lab in enumerate(all_labels)}
        def _remap(arr):
            if arr is None: return None
            return np.vectorize(mapping.get)(arr).astype(int)
        ytr = _remap(ytr); yval = _remap(yval); yts = _remap(yts)
    num_class = int(all_labels.size)
    ybin_tr  = _binary_mapping(ytr, num_class)
    ybin_val = _binary_mapping(yval, num_class) if yval is not None else None
    ybin_ts  = _binary_mapping(yts,  num_class) if yts  is not None else None

    # ---- populations & indices
    num_pop, pop_idx = _infer_num_pop(gp, xtr, cfg.user.pop_idx, cfg.runcontrol.num_pop)

    # ---- initial population shape (length only; detailed content checked later in engine)
    if cfg.user.initial_population is not None:
        init = cfg.user.initial_population
        if not isinstance(init, (list, tuple)) or len(init) != num_pop:
            gp.error("The length of user.initial_population must equal num_pop.", ValueError)

    # ---- runcontrol scalars
    rc = cfg.runcontrol
    pop_size        = _ensure_int(gp, "runcontrol.pop_size", rc.pop_size)
    generations     = _ensure_int(gp, "runcontrol.generations", rc.generations)
    batch_job       = _ensure_int(gp, "runcontrol.batch_job", rc.batch_job)
    stallgen        = _ensure_int(gp, "runcontrol.stallgen", rc.stallgen)
    rc_verbose      = _ensure_int(gp, "runcontrol.verbose", rc.verbose)
    savefreq        = _ensure_int(gp, "runcontrol.savefreq", rc.savefreq)
    quiet           = _ensure_bool(gp, "runcontrol.quiet", rc.quiet)
    useparallel     = _ensure_bool(gp, "runcontrol.useparallel", rc.useparallel)
    n_jobs          = _ensure_int(gp, "runcontrol.n_jobs", rc.n_jobs)
    usecache        = _ensure_bool(gp, "runcontrol.usecache", rc.usecache)
    minimisation    = _ensure_bool(gp, "runcontrol.minimisation", rc.minimisation)
    tolfit          = _ensure_float(gp, "runcontrol.tolfit", rc.tolfit)
    plotfitness     = _ensure_bool(gp, "runcontrol.plotfitness", rc.plotfitness)
    plotrankall     = _ensure_bool(gp, "runcontrol.plotrankall", rc.plotrankall)
    plotrankbest    = _ensure_bool(gp, "runcontrol.plotrankbest", rc.plotrankbest)
    plotsavefig     = _ensure_bool(gp, "runcontrol.plotsavefig", rc.plotsavefig)
    plotlive        = _ensure_bool(gp, "runcontrol.plotlive", rc.plotlive)
    plotbackend     = _validate_backend(gp, rc.plotbackend)
    plotformat      = _validate_plot_formats(gp, rc.plotformat)
    plotfolder      = _ensure_dir(gp, "runcontrol.plotfolder", rc.plotfolder, default_to_cwd=True)
    resultfolder    = _ensure_dir(gp, "runcontrol.resultfolder", rc.resultfolder, default_to_cwd=True)
    rng             = _init_rng(gp, rc.random_state)
    rc_random_state = rc.random_state

    agg_method = rc.agg_method
    if num_pop > 1 and (agg_method is None or str(agg_method).strip() == ""):
        agg_method = "Ensemble"
    elif not isinstance(agg_method, str):
        gp.error("'runcontrol.agg_method' must be a string.", ValueError)

    # ---- selection (broadcast to len=num_pop)
    sel = cfg.selection
    tournament_size   = _as_list_n(gp, "selection.tournament_size", sel.tournament_size, num_pop,
                                   validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e))
    elite_fraction    = _as_list_n(gp, "selection.elite_fraction", sel.elite_fraction, num_pop,
                                   validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    elite_fraction_en = _as_list_n(gp, "selection.elite_fraction_ensemble", sel.elite_fraction_ensemble, num_pop,
                                   validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    lex_pressure      = _as_list_n(gp, "selection.tournament_lex_pressure", sel.tournament_lex_pressure, num_pop,
                                   validator=lambda gp_, n_, e: _ensure_bool(gp_, n_, e))
    p_pareto          = _as_list_n(gp, "selection.tournament_p_pareto", sel.tournament_p_pareto, num_pop,
                                   validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    p_ensemble        = _as_list_n(gp, "selection.p_ensemble", sel.p_ensemble, num_pop,
                                   validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))

    # ---- constants
    ndc = cfg.nodes.const
    if ndc.about != "Ephemeral random constants":
        diags.warn(f"'nodes.const.about' is {ndc.about!r}; expected 'Ephemeral random constants'. Using provided value.")
    p_ERC   = _as_list_n(gp, "nodes.const.p_ERC", ndc.p_ERC, num_pop,
                         validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    p_int   = _as_list_n(gp, "nodes.const.p_int", ndc.p_int, num_pop,
                         validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    # range: [min,max] or list-of-pairs
    raw_cr  = ndc.range
    if _is_range_pair(raw_cr):
        pair = _validate_range_pair(gp, "nodes.const.range", raw_cr)
        const_range = [pair for _ in range(num_pop)]
    elif isinstance(raw_cr, (list, tuple)) and raw_cr and all(_is_range_pair(x) for x in raw_cr):
        if len(raw_cr) != num_pop:
            gp.error("'nodes.const.range' list length must equal num_pop.", ValueError)
        const_range = [_validate_range_pair(gp, "nodes.const.range", pair) for pair in raw_cr]
    else:
        gp.error("'nodes.const.range' must be [min,max] or a list of such pairs.", ValueError)
    num_dec_places = _as_list_n(gp, "nodes.const.num_dec_places", ndc.num_dec_places, num_pop,
                                validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e))

    # ---- functions
    ndf = cfg.nodes.functions
    fn_names_param = ndf.name
    if isinstance(fn_names_param, (list, tuple)) and fn_names_param and all(isinstance(x, str) for x in fn_names_param):
        fn_names = [list(fn_names_param)] * num_pop
    elif isinstance(fn_names_param, (list, tuple)) and fn_names_param and all(isinstance(x, (list, tuple)) for x in fn_names_param):
        if len(fn_names_param) != num_pop:
            gp.error("'nodes.functions.name' must have length num_pop when given per-pop.", ValueError)
        fn_names = [list(map(str, sub)) for sub in fn_names_param]
    else:
        gp.error("'nodes.functions.name' must be a list[str] or list[list[str]].", ValueError)
    functions_function = []
    functions_arity = []
    for names in fn_names:
        fdict = {}; ar = []
        for nm in names:
            fn, nargs = _import_operator_function(gp, nm)
            fdict[nm] = fn; ar.append(int(nargs))
        functions_function.append(fdict)
        functions_arity.append(ar)
    functions_active = [[1 for _ in names] for names in fn_names]

    # ---- operator
    op = cfg.operator
    p_mutate = _as_list_n(gp, "operator.p_mutate", op.p_mutate, num_pop,
                          validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    p_cross  = _as_list_n(gp, "operator.p_cross", op.p_cross, num_pop,
                          validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    p_direct = _as_list_n(gp, "operator.p_direct", op.p_direct, num_pop,
                          validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    # mutate_par: vector-of-6 or list-of-vectors
    def _is_mutvec(e) -> bool:
        return isinstance(e, (list, tuple)) and len(e) == 6 and all(_is_float(x) for x in e)
    def _validate_mutvec(gp_, name_, e):
        if not _is_mutvec(e):
            gp_.error(f"'{name_}' must be a list of six non-negative probabilities.", ValueError)
        vec = [float(x) for x in e]
        if any(x < 0.0 for x in vec):
            gp_.error(f"'{name_}' entries must be >= 0.", ValueError)
        s = sum(vec)
        if s <= 0:
            gp_.error(f"'{name_}' sum must be > 0.", ValueError)
        if abs(s - 1.0) > 1e-6:
            diags.warn(f"'{name_}' sums to {s:.6f}; normalizing to 1.0.")
            vec = [x / s for x in vec]
        return vec
    raw_mp = op.mutate_par
    if _is_mutvec(raw_mp):
        vec = _validate_mutvec(gp, "operator.mutate_par", raw_mp)
        mutate_par = [vec for _ in range(num_pop)]
    elif isinstance(raw_mp, (list, tuple)) and raw_mp and all(_is_mutvec(x) for x in raw_mp):
        if len(raw_mp) != num_pop:
            gp.error("'operator.mutate_par' list length must equal num_pop.", ValueError)
        mutate_par = [_validate_mutvec(gp, "operator.mutate_par", v) for v in raw_mp]
    else:
        gp.error("'operator.mutate_par' must be a 6-float vector or a list of such vectors.", ValueError)
    mutate_par_cumsum = [np.cumsum(v).tolist() for v in mutate_par]
    mutate_gaussian_std = _as_list_n(gp, "operator.mutate_gaussian_std", op.mutate_gaussian_std, num_pop,
                                     validator=lambda gp_, n_, e: _ensure_float(gp_, n_, e))

    # ---- gene
    ge = cfg.gene
    gene_p_cross_hi  = _as_list_n(gp, "gene.p_cross_hi", ge.p_cross_hi, num_pop,
                                  validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    gene_hi_cross_rate = _as_list_n(gp, "gene.hi_cross_rate", ge.hi_cross_rate, num_pop,
                                    validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    gene_multigene   = _as_list_n(gp, "gene.multigene", ge.multigene, num_pop,
                                  validator=lambda gp_, n_, e: _ensure_bool(gp_, n_, e))
    gene_max_genes   = _as_list_n(gp, "gene.max_genes", ge.max_genes, num_pop,
                                  validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e))

    # ---- tree
    tr = cfg.tree
    tree_build_method = _as_list_n(gp, "tree.build_method", tr.build_method, num_pop,
                                   validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e))
    def _val_nodes(gp_, name_, e):
        if isinstance(e, (int, np.integer)):
            return int(e)
        if e in (np.inf, float("inf")):
            return float("inf")
        gp_.error(f"'{name_}' must be an int or np.inf.", ValueError)
    tree_max_nodes = _as_list_n(gp, "tree.max_nodes", tr.max_nodes, num_pop, validator=_val_nodes)
    tree_max_depth = _as_list_n(gp, "tree.max_depth", tr.max_depth, num_pop,
                                validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e))
    tree_max_mutate_depth = _as_list_n(gp, "tree.max_mutate_depth", tr.max_mutate_depth, num_pop,
                                       validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e))

    # ---- fitness
    fit = cfg.fitness
    fitness_terminate = _ensure_bool(gp, "fitness.terminate", fit.terminate)
    cm = _ensure_int(gp, "fitness.complexityMeasure", fit.complexityMeasure)
    if cm not in (0, 1):
        gp.error("'fitness.complexityMeasure' must be 0 or 1.", ValueError)

    # ---- softmax
    sm = cfg.softmax
    opt_types   = _broadcast_optimizer_types(gp, sm.optimizer_type, num_pop)
    init_list   = _as_list_n(gp, "softmax.initializer", sm.initializer, num_pop,
                             validator=lambda gp_, n_, e: e if isinstance(e, str) and e in
                             {"glorot_uniform", "he_normal", "random_normal"} else gp_.error(
                                 "'softmax.initializer' must be one of "
                                 "['glorot_uniform','he_normal','random_normal']", ValueError))
    def _val_reg(gp_, name_, e):
        if e is None: return None
        if isinstance(e, str) and e in {"l1", "l2", "hybrid"}: return e
        gp_.error("'softmax.regularization' must be None|'l1'|'l2'|'hybrid'.", ValueError)
    reg_list    = _as_list_n(gp, "softmax.regularization", sm.regularization, num_pop, validator=_val_reg)
    reg_rate    = _as_list_n(gp, "softmax.regularization_rate", sm.regularization_rate, num_pop,
                             validator=lambda gp_, n_, e: _ensure_float(gp_, n_, e))
    batch_size  = _as_list_n(gp, "softmax.batch_size", sm.batch_size, num_pop,
                             validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e))
    epochs      = _as_list_n(gp, "softmax.epochs", sm.epochs, num_pop,
                             validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e))
    patience    = _as_list_n(gp, "softmax.patience", sm.patience, num_pop,
                             validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e))
    buffer_size = _as_list_n(gp, "softmax.buffer_size", sm.buffer_size, num_pop,
                             validator=lambda gp_, n_, e: (None if e is None else _ensure_int(gp_, n_, e)))
    shuffle     = _as_list_n(gp, "softmax.shuffle", sm.shuffle, num_pop,
                             validator=lambda gp_, n_, e: _ensure_bool(gp_, n_, e))
    sm_verbose  = _as_list_n(gp, "softmax.verbose", sm.verbose, num_pop,
                             validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e))
    opt_params  = _broadcast_optimizer_params(gp, opt_types, sm.optimizer_param, num_pop, diags)

    # warn if 'hybrid' but single scalar rate provided by user
    for i, (r, rr) in enumerate(zip(reg_list, reg_rate)):
        if r == "hybrid" and not (isinstance(sm.regularization_rate, (list, tuple)) and
                                  isinstance(sm.regularization_rate[0], (list, tuple))):
            diags.warn(f"softmax.regularization is 'hybrid' for pop {i}, "
                       f"but regularization_rate is scalar; consider [l1_rate, l2_rate].")

    # ---- finalize dictionaries as engine expects
    config = {
        "runcontrol": {
            "num_pop": num_pop,
            "pop_size": pop_size,
            "num_class": num_class,
            "num_generations": generations,
            "stallgen": stallgen,
            "tolfit": tolfit,
            "verbose": rc_verbose,
            "savefreq": savefreq,
            "quiet": quiet,
            "parallel": {"useparallel": useparallel, "n_jobs": n_jobs, "batch_job": batch_job},
            "usecache": usecache,
            "minimisation": minimisation,
            "plot": {
                "fitness": plotfitness,
                "rankall": plotrankall,
                "rankbest": plotrankbest,
                "savefig": plotsavefig,
                "live": plotlive,
                "backend": plotbackend,
                "format": plotformat,
                "folder": plotfolder,
            },
            "track_individuals": _ensure_bool(gp, "runcontrol.track_individuals", rc.track_individuals),
            "resultfolder": resultfolder,
            "agg_method": agg_method,
            "random_state": rc_random_state,
            "RNG": rng,
        },
        "selection": {
            "tournament_size": tournament_size,
            "elite_fraction": elite_fraction,
            "elite_fraction_ensemble": elite_fraction_en,
            "tournament_lex_pressure": lex_pressure,
            "tournament_p_pareto": p_pareto,
            "p_ensemble": p_ensemble,
        },
        "nodes": {
            "const": {
                "about": ndc.about,
                "p_ERC": p_ERC,
                "p_int": p_int,
                "range": const_range,
                "num_dec_places": num_dec_places,
            },
            "functions": {
                "name": fn_names,
                "function": functions_function,
                "arity": functions_arity,
                "active": functions_active,
            },
        },
        "operator": {
            "p_mutate": p_mutate,
            "p_cross": p_cross,
            "p_direct": p_direct,
            "mutate_par": mutate_par,
            "mutate_par_cumsum": [np.cumsum(v).tolist() for v in mutate_par],
            "mutate_gaussian_std": mutate_gaussian_std,
        },
        "gene": {
            "p_cross_hi": gene_p_cross_hi,
            "hi_cross_rate": gene_hi_cross_rate,
            "multigene": gene_multigene,
            "max_genes": gene_max_genes,
        },
        "tree": {
            "build_method": tree_build_method,
            "max_nodes": tree_max_nodes,
            "max_depth": tree_max_depth,
            "max_mutate_depth": tree_max_mutate_depth,
        },
        "fitness": {
            "fitfun": 1,  # placeholder for custom fit function wiring
            "terminate": fitness_terminate,
            "complexityMeasure": cm,
        },
        "softmax": {
            "optimizer_type": opt_types,
            "optimizer_param": opt_params,
            "initializer": init_list,
            "regularization": reg_list,
            "regularization_rate": reg_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "buffer_size": buffer_size,
            "shuffle": shuffle,
            "verbose": sm_verbose,
        },
    }

    userdata = {
        "name": str(cfg.user.name),
        "stats": _ensure_bool(gp, "user.stats", cfg.user.stats),
        "user_fcn": None,  # currently unused
        "pop_idx": [list(grp) for grp in pop_idx],
        "num_class": num_class,
        "xtrain": xtr.copy(),
        "ytrain": ytr.copy(),
        "xval": None if xval is None else xval.copy(),
        "yval": None if yval is None else yval.copy(),
        "xtest": None if xts is None else xts.copy(),
        "ytest": None if yts is None else yts.copy(),
        "ybinarytrain": ybin_tr.copy(),
        "ybinaryval": None if ybin_val is None else ybin_val.copy(),
        "ybinarytest": None if ybin_ts is None else ybin_ts.copy(),
        "initial_population": None if cfg.user.initial_population is None else cfg.user.initial_population,
    }

    if return_diagnostics:
        return config, userdata, diags
    return config, userdata
