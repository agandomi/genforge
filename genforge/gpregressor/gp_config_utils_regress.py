# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
from __future__ import annotations
from typing import Any, List, Tuple, Dict, Optional
import os, importlib, inspect
import numpy as np

from .gp_config_data_regress import (
    RegressorConfig, Diagnostics,
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
            "Using legacy np.random.RandomState; consider switching to np.random.default_rng()."
        )
        return np.random.Generator(np.random.MT19937(random_state.randint(0, 2**32)))

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

def _ensure_int(gp, name, v, *, min_val: Optional[int] = None):
    if not _is_int(v):
        gp.error(f"'{name}' must be an integer; got {type(v).__name__}={v!r}", ValueError)
    vi = int(v)
    if min_val is not None and vi < min_val:
        gp.error(f"'{name}' must be >= {min_val}; got {vi}", ValueError)
    return vi

def _ensure_float(gp, name, v, *, min_val: Optional[float] = None, max_val: Optional[float] = None, strict_pos: bool = False):
    if not _is_float(v):
        gp.error(f"'{name}' must be a float; got {type(v).__name__}={v!r}", ValueError)
    vf = float(v)
    if strict_pos and vf <= 0:
        gp.error(f"'{name}' must be > 0; got {vf}", ValueError)
    if min_val is not None and vf < min_val:
        gp.error(f"'{name}' must be >= {min_val}; got {vf}", ValueError)
    if max_val is not None and vf > max_val:
        gp.error(f"'{name}' must be <= {max_val}; got {vf}", ValueError)
    return vf

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
    """
    Broadcast scalar → len n, or validate list length. Optionally per-element validator.
    For nested lists (like l1_ratio grid), pass allow_scalar=False and do custom handling.
    """
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

# ------------------------- operator functions import -------------------------

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

# ------------------------- ElasticNetCV helpers -------------------------

def _broadcast_l1_ratio(gp, value, n):
    """
    Accept:
      - float -> per-pop [float]
      - list[float] -> same grid used for all populations
      - list[list[float]] -> per-pop grids (len == n)
    Validate each float in [0,1].
    """
    def _ensure_ratio(gp_, name_, x):
        return _ensure_prob01(gp_, name_, x, inclusive=True)

    if isinstance(value, (float, int, np.floating, np.integer)):
        grid = [_ensure_ratio(gp, "linregression.l1_ratio", value)]
        return [grid for _ in range(n)]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            gp.error("'linregression.l1_ratio' must not be empty.", ValueError)
        if all(isinstance(x, (float, int, np.floating, np.integer)) for x in value):
            grid = [_ensure_ratio(gp, "linregression.l1_ratio", x) for x in value]
            return [grid for _ in range(n)]
        if all(isinstance(x, (list, tuple)) for x in value):
            if len(value) != n:
                gp.error("'linregression.l1_ratio' must have length num_pop when given per-pop.", ValueError)
            out = []
            for i, sub in enumerate(value):
                if len(sub) == 0:
                    gp.error(f"'linregression.l1_ratio[{i}]' must not be empty.", ValueError)
                out.append([_ensure_ratio(gp, f"linregression.l1_ratio[{i}]", x) for x in sub])
            return out
    gp.error("'linregression.l1_ratio' must be a float, list[float], or list[list[float]].", ValueError)

def _broadcast_alpha_list(gp, value, n):
    """
    Accept None, list[float], or list[list[float]].
    Each alpha must be > 0.  Returns per-pop list (None or list).
    """
    def _ensure_alpha(gp_, name_, x):
        return _ensure_float(gp_, name_, x, strict_pos=True)

    if value is None:
        return [None for _ in range(n)]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            gp.error("'linregression.alphas' must not be empty when provided.", ValueError)
        if all(isinstance(x, (float, int, np.floating, np.integer)) for x in value):
            grid = [_ensure_alpha(gp, "linregression.alphas", x) for x in value]
            return [grid for _ in range(n)]
        if all(isinstance(x, (list, tuple)) for x in value):
            if len(value) != n:
                gp.error("'linregression.alphas' must have length num_pop when given per-pop.", ValueError)
            out = []
            for i, sub in enumerate(value):
                if len(sub) == 0:
                    gp.error(f"'linregression.alphas[{i}]' must not be empty.", ValueError)
                out.append([_ensure_alpha(gp, f"linregression.alphas[{i}]", x) for x in sub])
            return out
    gp.error("'linregression.alphas' must be None, list[float], or list[list[float]].", ValueError)

def _broadcast_simple(gp, name, value, n, *, validator):
    """Broadcast scalar or list length==n with the provided validator per element."""
    return _as_list_n(gp, name, value, n, validator=lambda gp_, nm, e: validator(gp_, nm, e))

def _validate_selection(gp, name, v):
    if not isinstance(v, str):
        gp.error(f"'{name}' must be a string.", ValueError)
    v = v.lower().strip()
    if v not in {"cyclic", "random"}:
        gp.error(f"'{name}' must be 'cyclic' or 'random'; got {v!r}", ValueError)
    return v

def _validate_n_jobs(gp, name, v):
    if v is None:
        return None
    return _ensure_int(gp, name, v, min_val=1)

# ------------------------- Resolve regressor config -------------------------

def resolve_regressor_config(gp, cfg: RegressorConfig, *, return_diagnostics: bool = False):
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

    # ---- populations & indices
    num_pop, pop_idx = _infer_num_pop(gp, xtr, cfg.user.pop_idx, cfg.runcontrol.num_pop)

    # ---- initial population length (content checked later by engine)
    if cfg.user.initial_population is not None:
        init = cfg.user.initial_population
        if not isinstance(init, (list, tuple)) or len(init) != num_pop:
            gp.error("The length of user.initial_population must equal num_pop.", ValueError)

    # ---- runcontrol scalars
    rc = cfg.runcontrol
    pop_size        = _ensure_int(gp, "runcontrol.pop_size", rc.pop_size, min_val=1)
    generations     = _ensure_int(gp, "runcontrol.generations", rc.generations, min_val=1)
    batch_job       = _ensure_int(gp, "runcontrol.batch_job", rc.batch_job, min_val=1)
    stallgen        = _ensure_int(gp, "runcontrol.stallgen", rc.stallgen, min_val=0)
    adaptgen        = _ensure_int(gp, "runcontrol.adaptgen", rc.adaptgen, min_val=0)
    adaptinject     = _ensure_bool(gp, "runcontrol.adaptinject", rc.adaptinject)
    rc_verbose      = _ensure_int(gp, "runcontrol.verbose", rc.verbose, min_val=0)
    savefreq        = _ensure_int(gp, "runcontrol.savefreq", rc.savefreq, min_val=0)
    quiet           = _ensure_bool(gp, "runcontrol.quiet", rc.quiet)
    useparallel     = _ensure_bool(gp, "runcontrol.useparallel", rc.useparallel)
    n_jobs          = _ensure_int(gp, "runcontrol.n_jobs", rc.n_jobs, min_val=1)
    usecache        = _ensure_bool(gp, "runcontrol.usecache", rc.usecache)
    minimisation    = _ensure_bool(gp, "runcontrol.minimisation", rc.minimisation)
    tolfit          = _ensure_float(gp, "runcontrol.tolfit", rc.tolfit, strict_pos=True)
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
        agg_method = "Regression"
    elif not isinstance(agg_method, str):
        gp.error("'runcontrol.agg_method' must be a string.", ValueError)

    # ---- selection (broadcast to len=num_pop)
    sel = cfg.selection
    tournament_size   = _as_list_n(gp, "selection.tournament_size", sel.tournament_size, num_pop,
                                   validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e, min_val=1))
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
                                validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e, min_val=0))

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
                                     validator=lambda gp_, n_, e: _ensure_float(gp_, n_, e, strict_pos=True))

    # ---- gene
    ge = cfg.gene
    gene_p_cross_hi  = _as_list_n(gp, "gene.p_cross_hi", ge.p_cross_hi, num_pop,
                                  validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    gene_hi_cross_rate = _as_list_n(gp, "gene.hi_cross_rate", ge.hi_cross_rate, num_pop,
                                    validator=lambda gp_, n_, e: _ensure_prob01(gp_, n_, e))
    gene_multigene   = _as_list_n(gp, "gene.multigene", ge.multigene, num_pop,
                                  validator=lambda gp_, n_, e: _ensure_bool(gp_, n_, e))
    gene_max_genes   = _as_list_n(gp, "gene.max_genes", ge.max_genes, num_pop,
                                  validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e, min_val=1))

    # ---- tree
    tr = cfg.tree
    tree_build_method = _as_list_n(gp, "tree.build_method", tr.build_method, num_pop,
                                   validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e, min_val=0))
    def _val_nodes(gp_, name_, e):
        if isinstance(e, (int, np.integer)):
            return int(e)
        if e in (np.inf, float("inf")):
            return float("inf")
        gp_.error(f"'{name_}' must be an int or np.inf.", ValueError)
    tree_max_nodes = _as_list_n(gp, "tree.max_nodes", tr.max_nodes, num_pop, validator=_val_nodes)
    tree_max_depth = _as_list_n(gp, "tree.max_depth", tr.max_depth, num_pop,
                                validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e, min_val=1))
    tree_max_mutate_depth = _as_list_n(gp, "tree.max_mutate_depth", tr.max_mutate_depth, num_pop,
                                       validator=lambda gp_, n_, e: _ensure_int(gp_, n_, e, min_val=1))

    # ---- fitness
    fit = cfg.fitness
    fitness_terminate = _ensure_bool(gp, "fitness.terminate", fit.terminate)
    cm = _ensure_int(gp, "fitness.complexityMeasure", fit.complexityMeasure)
    if cm not in (0, 1):
        gp.error("'fitness.complexityMeasure' must be 0 or 1.", ValueError)

    # ---- ElasticNetCV (per-pop broadcast/validation)
    lr = cfg.linregression
    l1_ratio   = _broadcast_l1_ratio(gp, lr.l1_ratio, num_pop)  # per-pop list[list[float]]
    alphas     = _broadcast_alpha_list(gp, lr.alphas, num_pop)  # per-pop list[None|list[float]]
    n_alphas   = _broadcast_simple(gp, "linregression.n_alphas", lr.n_alphas, num_pop,
                                   validator=lambda gp_, nm, e: _ensure_int(gp_, nm, e, min_val=1))
    # scikit allows eps in (0,1); enforce strictly positive and <1 for sensible grids
    eps        = _broadcast_simple(gp, "linregression.eps", lr.eps, num_pop,
                                   validator=lambda gp_, nm, e: _ensure_float(gp_, nm, e, min_val=1e-12, max_val=1.0))
    fit_intercept = _broadcast_simple(gp, "linregression.fit_intercept", lr.fit_intercept, num_pop,
                                      validator=_ensure_bool)
    copy_x     = _broadcast_simple(gp, "linregression.copy_x", lr.copy_x, num_pop,
                                   validator=_ensure_bool)
    max_iter   = _broadcast_simple(gp, "linregression.max_iter", lr.max_iter, num_pop,
                                   validator=lambda gp_, nm, e: _ensure_int(gp_, nm, e, min_val=1))
    tol        = _broadcast_simple(gp, "linregression.tol", lr.tol, num_pop,
                                   validator=lambda gp_, nm, e: _ensure_float(gp_, nm, e, strict_pos=True))
    cv         = _broadcast_simple(gp, "linregression.cv", lr.cv, num_pop,
                                   validator=lambda gp_, nm, e: _ensure_int(gp_, nm, e, min_val=2))
    njobs      = _as_list_n(gp, "linregression.n_jobs", lr.n_jobs, num_pop,
                            validator=_validate_n_jobs)
    verbose    = _broadcast_simple(gp, "linregression.verbose", lr.verbose, num_pop,
                                   validator=lambda gp_, nm, e: _ensure_int(gp_, nm, e, min_val=0))
    positive   = _broadcast_simple(gp, "linregression.positive", lr.positive, num_pop,
                                   validator=_ensure_bool)
    selection  = _broadcast_simple(gp, "linregression.selection", lr.selection, num_pop,
                                   validator=_validate_selection)

    # ---- finalize dictionaries as engine expects
    config = {
        "runcontrol": {
            "num_pop": num_pop,
            "pop_size": pop_size,
            "num_generations": generations,
            "stallgen": stallgen,
            "adaptgen": adaptgen,
            "adaptinject": adaptinject,
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
            "mutate_par_cumsum": mutate_par_cumsum,
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
        "linregression": {
            # Each field below is broadcast per-pop:
            "l1_ratio": l1_ratio,           # List[List[float]]
            "alphas": alphas,               # List[None|List[float]]
            "n_alphas": n_alphas,           # List[int]
            "eps": eps,                     # List[float]
            "fit_intercept": fit_intercept, # List[bool]
            "copy_x": copy_x,               # List[bool]
            "max_iter": max_iter,           # List[int]
            "tol": tol,                     # List[float]
            "cv": cv,                       # List[int]
            "n_jobs": njobs,                # List[None|int]
            "verbose": verbose,             # List[int]
            "positive": positive,           # List[bool]
            "selection": selection,         # List[str]
        },
    }

    userdata = {
        "name": str(cfg.user.name),
        "stats": _ensure_bool(gp, "user.stats", cfg.user.stats),
        "user_fcn": None,  # currently unused
        "pop_idx": [list(grp) for grp in pop_idx],
        "xtrain": xtr.copy(),
        "ytrain": ytr.copy(),
        "xval": None if xval is None else xval.copy(),
        "yval": None if yval is None else yval.copy(),
        "xtest": None if xts is None else xts.copy(),
        "ytest": None if yts is None else yts.copy(),
        "initial_population": None if cfg.user.initial_population is None else cfg.user.initial_population,
    }

    if return_diagnostics:
        return config, userdata, diags
    return config, userdata
