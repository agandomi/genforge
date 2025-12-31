# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
import numpy as np

# ----------------------------- Diagnostics -----------------------------

@dataclass
class Diagnostics:
    """Non-fatal config messages to emit via gp.warning after resolve()."""
    warnings: List[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(str(msg))

# ----------------------------- Sections -----------------------------

@dataclass
class WarningCfg:
    warnings_mode: str = "default"                 # "default"|"ignore"|"error"
    log_capture_external_warnings: bool = False    # keep False so 3rd-party warnings stay untouched

@dataclass
class LogCfg:
    log_level: str = "INFO"                        # "DEBUG"|"INFO"|"WARNING"|"ERROR"|"CRITICAL"
    log_to_file: bool = False
    log_file: Optional[str] = None     # None -> use <runname>.log

# runcontrol_*  ->  runcontrol: <field>
@dataclass
class Runcontrol:
    num_pop: Optional[int] = None               # Number of populations (determined by idx from user input)
    agg_method: Optional[str] = "Regression"    # The Aggreagation Method
    pop_size: int = 25                          # Population size
    generations: int = 150                      # Number of generations
    batch_job: int = 5                          # number of tasks assigned to each cpu core
    stallgen: int = 20                          # Terminate if fitness doesn't change after stallgen generations
    adaptgen: int = 15                          # Newly added for adaptive orthogonal symbolic injection
    adaptinject: bool = False                     # For adaptive orthogonal symbolic injection
    verbose: int = 1                            # The generation frequency with which results are printed to Console
    savefreq: int = 0                           # The generation frequency to save the results
    quiet: bool = False                         # If true, then GP runs with no console output
    useparallel: bool = False                   # true to manually enable parallel CPU fitness evals (requires multiprocessing)
    n_jobs: int = 1                             # if parallel fitness evals enabled, this is the number of "CPUs" to use
    usecache: bool = True                       # fitness caching: used when copying individuals in a gen
    minimisation: bool = True                   # True if the problem is minimization and False if it is maximization
    tolfit: float = 1e-9                        # Tolfit means if fitness doesn't change as much as tolfit it is considered not improving
    plotfitness: bool = False                    # plot the fitness
    plotrankall: bool = False                    # plot all individual rankings
    plotrankbest: bool = False                   # plot the best ensemble and individual rankings
    plotformat: Any = "png"                     # str or List[str] plot format
    plotfolder: Optional[str] = None            # None -> CWD (created if needed) plot folder
    plotsavefig: bool = False                    # save figures
    plotlive: bool = False                       # set True for a live window, False for save-only
    plotbackend: str = "auto"                   # 'auto'|'QtAgg'|'TkAgg'|'MacOSX'|'Agg' 
    track_individuals: bool = False             # Track the individuals across generations
    resultfolder: Optional[str] = None          # None -> CWD (created if needed)
    random_state: Optional[Any] = None          # None, int, class, random seed generator

# selection_* -> selection: <field>
@dataclass
class Selection:
    tournament_size: Any = 2                       # int or List[int], Tournament size
    elite_fraction: Any = 0.05                     # float or List[float], Elite fraction in isolated population
    elite_fraction_ensemble: Any = 0.05            # float or List[float], Elite fraction for all populations
    tournament_lex_pressure: Any = True            # bool or List[bool], set to true to use Sean Luke's et al.'s lexographic selection pressure during regular tournament selection
    tournament_p_pareto: Any = 0.0                 # float or List[float], probability that a pareto tournament will be used for any given selection event.
    p_ensemble: Any = 0.0                          # float or List[float], probability of using ensemble fitness for selection

# nodes.const / nodes.functions
@dataclass
class NodesConst:
    about: str = "Ephemeral random constants"      # constant nodes generation method
    p_ERC: Any = 0.1                               # float or List[float], probability of generating an ERC when creating a leaf node
    p_int: Any = 0.0                               # float or List[float], probability of generating an integer ERC
    range: Any = field(default_factory=lambda: [-10, 10])                         # [min, max] or List[[min, max], ...], ERC range 
    num_dec_places: Any = 4                        # int or List[int], decimal places

@dataclass
class NodesFunctions:
    name: Any = ("times", "minus", "plus")             # List[str] or List[List[str]], the functional nodes operators dictionary
    function: Optional[List[Dict[str, Any]]] = None    # filled by resolve(), the function handles
    arity: Optional[List[List[int]]] = None            # filled by resolve(), arity
    active: Optional[List[List[int]]] = None           # filled by resolve(), active

@dataclass
class Nodes:
    const: NodesConst = field(default_factory=NodesConst)
    functions: NodesFunctions = field(default_factory=NodesFunctions)

# operator_*
@dataclass
class Operator:
    p_mutate: Any = 0.14                                 # float or List[float],  Mutation probability 
    p_cross: Any = 0.84                                  # Crossover probability
    p_direct: Any = 0.02                                 # 
    mutate_par: Any = field(default_factory=lambda: [0.9, 0.05, 0.05, 0.0, 0.0, 0.0])    # 6-float vector or List[vector], probability of mutation from [any node, input, constant with guassian distribution]
    mutate_gaussian_std: Any = 0.1                       # float or List[float], for mutate_type 3 (constant perturbation): the standard deviation of the Gaussian used.

# gene_*
@dataclass
class Gene:
    p_cross_hi: Any = 0.2                          # float or List[float], probability of high level crossover
    hi_cross_rate: Any = 0.5                       # probability of any given gene being selected during high level crossover
    multigene: Any = True                          # bool or List[bool], Set to true if individuals can have multiple genes
    max_genes: Any = 5                             # int or List[int], Maximum number of genes per individual

# tree_*
@dataclass
class Tree:
    build_method: Any = 3                          # int or List[int], 3 = ramped half and half 
    max_nodes: Any = np.Inf                        # int|inf or List[...], Maximum nodes that a tree can have
    max_depth: Any = 4                             # int or List[int], Maximum depth of the trees
    max_mutate_depth: Any = 4                      # int or List[int], Maximum mutation depth in the trees 

# fitness_*
@dataclass
class Fitness:
    terminate: bool = False                         # true to enable early run termination on attaining a certain fitness value.
    complexityMeasure: int = 1                      # 0|1, 1 = expressional complexity 0 = number of nodes

# softmax_*  ->  softmax: <field>
@dataclass
class Linregression:
    l1_ratio: Any = field(default_factory=lambda: [1.0])  # list[float] or List[list[float]] The mixing between L1 and L2: 0.0 → Ridge, 1.0 → Lasso, values in between → Elastic Net.
    alphas: Any = None                             # None or list[float] or list[list[float]] Explicit list of alphas to try. If set, it overrides eps and n_alphas.
    n_alphas: Any = 100                            # int or List[int] Number of alpha values in the automatically generated grid (ignored if you pass alphas explicitly).
    eps: Any = 1e-3                                # float or list[float] When alphas=None, the alpha grid is generated on a log scale between alpha_max and alpha_min = eps * alpha_max.
    fit_intercept: Any = True                      # bool or List[bool] Whether to fit an intercept. If you’ve already centered your features/targets, you can set False.
    copy_x: Any = True                             # bool or List[bool] If True, X is copied; if False, it may be overwritten/centered in place (small perf win, be careful).
    max_iter: Any = 1000                           # int or List[int], Maximum coordinate-descent iterations per fit. Increase if you see convergence warnings.
    tol: Any = 1e-4                                # float or List[float] Optimization stopping tolerance (on the objective). Smaller → more precise, slower.
    cv: Any = 5                                    # int or list[int] crossfolds
    n_jobs: Any = None                             # None, int or list[int] for parallel jobs 
    verbose: Any = 0                               # int or list[int] Controls logging during fitting. 0 = silent, higher values = more messages.
    positive: Any = False                          # bool or list[bool] Constrain coefficients to be non-negative. (Use with caution—can raise training error if data aren’t scaled/centered suitably.)
    selection: Any = "cyclic"                      # str or list[str], "cyclic" or "random" Coordinate-descent sweep order. 'cyclic' is deterministic; 'random' can be faster on large problems

# userdata_*  ->  user: <field>
@dataclass
class User:
    name: str = "Example GP"                        # Name of the run
    stats: bool = True                              # update the stats
    user_fcn: Any = None                            # user function
    xtrain: Optional[np.ndarray] = None             # x train data
    ytrain: Optional[np.ndarray] = None             # y train data
    xval: Optional[np.ndarray] = None               # x vaildation data
    yval: Optional[np.ndarray] = None               # y validation data
    xtest: Optional[np.ndarray] = None              # x test data
    ytest: Optional[np.ndarray] = None              # y test data
    pop_idx: Any = None                             # None, List[int], or List[List[int]], the column index of x in multi-population
    initial_population: Any = None                  # None or List[per-pop individuals]

# ----------------------------- Top-level -----------------------------

@dataclass
class RegressorConfig:
    # Back-compat aliases: cfg.nodesconst <-> cfg.nodes.const
    @property
    def nodesconst(self):
        return self.nodes.const
    @nodesconst.setter
    def nodesconst(self, value):
        self.nodes.const = value

    # Back-compat aliases: cfg.nodesfunctions <-> cfg.nodes.functions
    @property
    def nodesfunctions(self):
        return self.nodes.functions
    @nodesfunctions.setter
    def nodesfunctions(self, value):
        self.nodes.functions = value
        
    runcontrol: Runcontrol = field(default_factory=Runcontrol)
    selection: Selection = field(default_factory=Selection)
    nodes: Nodes = field(default_factory=Nodes)
    operator: Operator = field(default_factory=Operator)
    gene: Gene = field(default_factory=Gene)
    tree: Tree = field(default_factory=Tree)
    fitness: Fitness = field(default_factory=Fitness)
    linregression: Linregression = field(default_factory=Linregression)
    user: User = field(default_factory=User)
    log: LogCfg = field(default_factory=LogCfg)
    warning: WarningCfg = field(default_factory=WarningCfg)

    # ---- Legacy adapter: build config from old flat parameters dict ----
    @classmethod
    def from_legacy(cls, params: Dict[str, Any]) -> "RegressorConfig":
        # Helper to pull with default if missing
        g = lambda k, d=None: params.get(k, d)

        # Build sections using legacy keys
        run = Runcontrol(
            num_pop=g("runcontrol_num_pop"),
            agg_method=g("runcontrol_agg_method", "Regression"),
            pop_size=g("runcontrol_pop_size", 25),
            generations=g("runcontrol_generations", 150),
            batch_job=g("runcontrol_batch_job", 5),
            stallgen=g("runcontrol_stallgen", 20),
            adaptgen=g("runcontrol_adaptgen", 15),
            adaptinject=g("runcontrol_adaptinject", True),
            verbose=g("runcontrol_verbose", 1),
            savefreq=g("runcontrol_savefreq", 0),
            quiet=g("runcontrol_quiet", False),
            useparallel=g("runcontrol_useparallel", False),
            n_jobs=g("runcontrol_n_jobs", 1),
            usecache=g("runcontrol_usecache", True),
            minimisation=g("runcontrol_minimisation", True),
            tolfit=g("runcontrol_tolfit", 1e-9),
            plotfitness=g("runcontrol_plotfitness", True),
            plotrankall=g("runcontrol_plotrankall", True),
            plotrankbest=g("runcontrol_plotrankbest", True),
            plotformat=g("runcontrol_plotformat", "png"),
            plotfolder=g("runcontrol_plotfolder", None),
            plotsavefig=g("runcontrol_plotsavefig", True),
            plotlive=g("runcontrol_plotlive", True),
            plotbackend=g("runcontrol_plotbackend", "auto"),
            track_individuals=g("runcontrol_track_individuals", False),
            resultfolder=g("runcontrol_resultfolder", None),
            random_state=g("runcontrol_random_state", None)
        )
        sel = Selection(
            tournament_size=g("selection_tournament_size", 2),
            elite_fraction=g("selection_elite_fraction", 0.05),
            elite_fraction_ensemble=g("selection_elite_fraction_ensemble", 0.05),
            tournament_lex_pressure=g("selection_tournament_lex_pressure", True),
            tournament_p_pareto=g("selection_tournament_p_pareto", 0.0),
            p_ensemble=g("selection_p_ensemble", 0.0),
        )
        ndc = NodesConst(
            about=g("const_about", "Ephemeral random constants"),
            p_ERC=g("const_p_ERC", 0.1),
            p_int=g("const_p_int", 0.0),
            range=g("const_range", [-10, 10]),
            num_dec_places=g("const_num_dec_places", 4),
        )
        ndf = NodesFunctions(
            name=g("functions_name", ["times", "minus", "plus"]),
        )
        nodes = Nodes(const=ndc, functions=ndf)
        op = Operator(
            p_mutate=g("operator_p_mutate", 0.14),
            p_cross=g("operator_p_cross", 0.84),
            p_direct=g("operator_p_direct", 0.02),
            mutate_par=g("operator_mutate_par", [0.9, 0.05, 0.05, 0, 0, 0]),
            mutate_gaussian_std=g("operator_mutate_gaussian_std", 0.1),
        )
        gene = Gene(
            p_cross_hi=g("gene_p_cross_hi", 0.2),
            hi_cross_rate=g("gene_hi_cross_rate", 0.5),
            multigene=g("gene_multigene", True),
            max_genes=g("gene_max_genes", 5),
        )
        tree = Tree(
            build_method=g("tree_build_method", 3),
            max_nodes=g("tree_max_nodes", np.Inf),
            max_depth=g("tree_max_depth", 4),
            max_mutate_depth=g("tree_max_mutate_depth", 4),
        )
        fit = Fitness(
            terminate=g("fitness_terminate", False),
            complexityMeasure=g("fitness_complexityMeasure", 1),
        )
        lnreg = Linregression(
            l1_ratio=g("linregression_l1_ratio", [1.0]),
            alphas=g("linregression_alphas", None),
            n_alphas=g("linregression_n_alphas", 100),
            eps=g("linregression_eps", 1e-3),
            fit_intercept=g("linregression_fit_intercept", True),
            copy_x=g("linregression_copy_x", True),
            max_iter=g("linregression_max_iter", 1000),
            tol=g("linregression_tol", 1e-4),
            cv=g("linregression_cv", 5),
            n_jobs=g("linregression_n_jobs", None),
            verbose=g("linregression_verbose", 0),
            positive=g("linregression_positive", False),
            selection=g("linregression_selection", "cyclic"),
        )
        user = User(
            name=g("userdata_name", "Example GP"),
            stats=g("userdata_stats", True),
            user_fcn=g("userdata_user_fcn", None),
            xtrain=g("userdata_xtrain"),
            ytrain=g("userdata_ytrain"),
            xval=g("userdata_xval"),
            yval=g("userdata_yval"),
            xtest=g("userdata_xtest"),
            ytest=g("userdata_ytest"),
            pop_idx=g("userdata_pop_idx", None),
            initial_population=g("userdata_initial_population", None),
        )
        log = LogCfg(
            log_level=g("log_level", "INFO"),
            log_to_file=g("log_to_file", False),
            log_file=g("log_file", None),
        )
        warn = WarningCfg(
            warnings_mode=g("warnings_mode", "default"),
            log_capture_external_warnings=g("log_capture_external_warnings", False),
        )
        return cls(runcontrol=run, selection=sel, nodes=nodes, operator=op,
                   gene=gene, tree=tree, fitness=fit, linregression=lnreg,
                   user=user, log=log, warning=warn)
