# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import warnings
from ..exceptions import GenForgeWarning
from .._log import get_logger, set_level, add_file_handler, close_handler, capture_external_warnings, log_exceptions
from .gp_config_data_class import ClassifierConfig
from .gp_config_class import gp_config

class gpclassifier:
    def __init__(self, cfg: ClassifierConfig | None = None, **legacy_params):
        # Did the caller provide both styles?
        had_cfg = cfg is not None
        had_legacy = bool(legacy_params)


        # Build the effective config (prefer cfg if provided; otherwise build from legacy; else defaults)
        if cfg is None:
            if had_legacy:
                # Requires ClassifierConfig.from_legacy(...) in your dataclass module
                cfg = ClassifierConfig.from_legacy(legacy_params)
            else:
                cfg = ClassifierConfig()

        self.cfg: ClassifierConfig = cfg

        # ---- initialize instance state (no gp.parameters)
        self.logger = {}
        self._file_handler = None
        self.config = {}
        self.userdata = {}
        self.state = {}
        self.fitness = {}
        self.individuals = {}
        self.track = {}
        self.info = {}
        self.cache = {}
        self.population = []
        self.plot = {}

        # Friendly run name
        self.runname = self.cfg.user.name

        # Set up logging/warnings BEFORE raising any error so it’s captured
        self._init_logging()

        # If both styles were provided, raise via self.error so it’s logged with stack
        if had_cfg and had_legacy:
            self.error("Provide either cfg=ClassifierConfig(...) OR legacy keyword arguments, not both.", TypeError)

        # Continue normal init
        self.information()
        self.configure()
        self.clearcache()

    # ---------------- logging & warnings ----------------

    def _init_logging(self) -> None:
        warn_cfg = self.cfg.warning
        log_cfg  = self.cfg.log
    
        self.logger = get_logger(f"genforge.gpclassifier.{id(self)}")
        set_level(self.logger, log_cfg.log_level)
    
        if log_cfg.log_to_file:
            import os, re
            # use user-provided path or derive from runname
            if log_cfg.log_file:
                log_path = log_cfg.log_file
            else:
                base = self.runname or "GenForge"
                safe = re.sub(r'[^A-Za-z0-9._ -]+', '_', str(base)).strip('_ ') or "GenForge"
                log_path = f"{safe}.log"
    
            # ensure parent dir exists (supports paths like "logs/myrun.log")
            parent = os.path.dirname(log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
    
            self._file_handler = add_file_handler(
                self.logger,
                log_path,
                mode="a",
                delay=True,
            )
            self.logger.info("Logging initialized for run %s → %s", self.runname, log_path)
    
        # keep external warnings off unless explicitly enabled
        capture_external_warnings(bool(warn_cfg.log_capture_external_warnings))

    def close_logs(self) -> None:
        close_handler(self.logger, getattr(self, "_file_handler", None))
        self._file_handler = None

    def warning(self, msg: str, *, stacklevel: int = 3) -> None:
        mode = self.cfg.warning.warnings_mode
        if getattr(self, "logger", None):
            self.logger.warning(msg, stack_info=True)
        if mode == "ignore":
            return
        elif mode == "error":
            raise RuntimeError(msg)
        warnings.warn(msg, GenForgeWarning, stacklevel=stacklevel)

    def error(self, msg: str, exc: type[Exception] = ValueError) -> None:
        if getattr(self, "logger", None):
            self.logger.error(msg, stack_info=True)
        raise exc(msg)

    # ---------------- lifecycle / wiring ----------------

    def information(self) -> None:
        from .._info import collect_package_info
        import datetime as _dt
        self.info.update(collect_package_info())
        self.info.setdefault("run", {})["name"] = self.cfg.user.name
        self.info["run"]["created_at"] = _dt.datetime.now().isoformat(timespec="seconds")

    @log_exceptions()
    def configure(self) -> None:
        # Resolve dataclass config to engine dicts
        gp_config(self)  # fills self.config, self.userdata from self.cfg

        # Wire engine based on agg method
        agg = self.config["runcontrol"]["agg_method"].lower()
        if agg == "ensemble":
            from .gp_individuals_ensemble_class import gp_individuals_ensemble
            from .gp_state_init_ensemble_class import gp_state_init_ensemble
            from .gp_fitness_init_class import gp_fitness_init
            gp_individuals_ensemble(self)
            gp_state_init_ensemble(self)
            gp_fitness_init(self)
        else:
            self.warning(f"Unknown agg_method '{agg}', using ensemble wiring.")

    @log_exceptions()
    def clearcache(self) -> None:
        if self.config['runcontrol']['usecache']:
            from .gp_cache_class import gp_cache
            gp_cache(self)

    @log_exceptions()
    def track_param(self) -> None:
        if self.config['runcontrol']['agg_method'].lower() == 'ensemble':
            from .gp_track_param_ensemble_class import gp_track_param_ensemble
            gp_track_param_ensemble(self)

    @log_exceptions()
    def build_pop(self) -> None:
        from .gp_popbuild_init import gp_popbuild_init
        from .gp_popbuild_class import gp_popbuild
        if self.state.get('generation', 0) == 0:
            gp_popbuild_init(self)
        else:
            gp_popbuild(self)

    @log_exceptions()
    def evalfitness(self) -> None:
        if self.config['runcontrol']['parallel']['useparallel']:
            from .gp_evalfitness_par_class import gp_evalfitness_par
            gp_evalfitness_par(self)
        else:
            from .gp_evalfitness_ord_class import gp_evalfitness_ord
            gp_evalfitness_ord(self)

    @log_exceptions()
    def updatestats(self) -> None:
        from .gp_updatestats_class import gp_updatestats
        gp_updatestats(self)

    @log_exceptions()
    def displaystats(self) -> None:
        from .gp_displaystats_class import gp_displaystats
        gp_displaystats(self)

    @log_exceptions()
    def plotstats(self) -> None:
        from .gp_plot_prepare_data import gp_plot_prepare_data
        gp_plot_prepare_data(self)

        live = bool(self.config['runcontrol']['plot'].get('live', False))
        backend_pref = self.config['runcontrol']['plot'].get('backend', 'auto')

        from .gp_plot_runtime import setup_matplotlib_backend
        _ = setup_matplotlib_backend(live=live, preferred=backend_pref)

        from .gp_plotfitness_class import gp_plotfitness
        from .gp_plotrankall_class import gp_plotrankall
        from .gp_plotrankbest_class import gp_plotrankbest
        if self.config['runcontrol']['plot']['fitness']:
            gp_plotfitness(self, live=live)
        if self.config['runcontrol']['plot']['rankall']:
            gp_plotrankall(self, live=live)
        if self.config['runcontrol']['plot']['rankbest']:
            gp_plotrankbest(self, live=live)

    @log_exceptions()
    def report(self, ensemble_row: int = 0, out_path: str = None, title: str = None) -> str:
        from .gp_html_report_class import write_ensemble_html
        import os, re
        run_name = self.userdata.get('name', self.cfg.user.name)
        if title is None:
            title = f"{run_name}: chosen ensemble #{ensemble_row}"
        safe_base = re.sub(r'[^A-Za-z0-9._-]+', '_', run_name).strip('_') or 'GenForge'
        default_filename = f"{safe_base}_ensemble_{ensemble_row}.html"
        if out_path is None:
            out_path = os.path.join(os.getcwd(), default_filename)
        return write_ensemble_html(self, ensemble_row=ensemble_row, out_path=out_path, title=title)
    
    @log_exceptions()
    def predict(self, X, *, ensemble_row: int = 0, return_proba: bool = False,
            mode: str = "ensemble", id_pop: int | None = None, id_ind: int | None = None):
        from .gp_predict_class import gp_predict
        """
        Predict with a chosen ensemble row (0-based) on a new 2D array X.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.
        ensemble_row : int, default=0
            Which ensemble row to use (0-based). Must be < len(self.individuals['ensemble_idx']).
        return_proba : bool, default=False
            If True, return class probabilities [n_samples, n_classes]; else return class indices [n_samples].
    
        Returns
        -------
        np.ndarray
            Probabilities or class indices depending on `return_proba`.
        """
        return gp_predict(self, X, ensemble_row=ensemble_row, return_proba=return_proba,
                      mode=mode, id_pop=id_pop, id_ind=id_ind)
    
    @log_exceptions()
    def plotcomplexity(self, mode: str = "all", id_pop: int | None = None,
                       savefig: bool | None = None, filename: str | None = None,
                       fmt: str | None = None, live: bool | None = None):
        # Match your other plotting entry-points re: backend selection
        from .gp_plot_runtime import setup_matplotlib_backend
        plot_cfg = self.config.get('runcontrol', {}).get('plot', {}) or {}
        live_final = plot_cfg.get('live', False) if live is None else bool(live)
        setup_matplotlib_backend(live=live_final, preferred=plot_cfg.get('backend', 'auto'))
    
        from .gp_plotcomplexity_class import gp_plotcomplexity
        return gp_plotcomplexity(self, mode=mode, id_pop=id_pop,
                                 savefig=savefig, filename=filename, fmt=fmt, live=live_final)

    @classmethod
    def initialize(cls, *args, **kwargs):
        """Back-compat: forwards to __init__ so you can pass cfg=... or legacy kwargs."""
        return cls(*args, **kwargs)

    @log_exceptions()
    def evolve(self) -> None:
        start_gen = self.state.get('generation', 0) + 1
        end_gen = self.config['runcontrol']['num_generations']
        for gen in range(start_gen, end_gen + 1):
            self.state['generation'] = gen
            self.track_param()
            self.build_pop()
            self.evalfitness()
            self.updatestats()
            self.clearcache()
            self.displaystats()
            self.plotstats()
            if self.state.get('terminate') or self.state.get('run_completed'):
                break
        if self._file_handler is not None:
            self.close_logs()
