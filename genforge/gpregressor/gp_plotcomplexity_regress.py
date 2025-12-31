# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def _ensure_palette(n):
    try:
        import seaborn as sns
        return sns.color_palette("tab10", n) if n <= 10 else sns.color_palette("tab20", n)
    except Exception:
        # Fallback: cycle Matplotlib defaults
        return [plt.cm.tab10(i % 10) for i in range(n)]

def _latexish():
    # Keep it LaTeX-like without requiring a TeX installation.
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    })

def _safe_plot_cfg(gp):
    plotcfg = gp.config.get("runcontrol", {}).get("plot", {}) or {}
    return {
        "live": bool(plotcfg.get("live", False)),
        "folder": plotcfg.get("folder", None),
        "savefig": bool(plotcfg.get("savefig", False)),
        "format": plotcfg.get("format", "png"),
    }

def _pick_fmt(fmt_cfg):
    # cfg may be a string like "png" or a list of fmts; pick the first
    if isinstance(fmt_cfg, (list, tuple)) and len(fmt_cfg) > 0:
        return str(fmt_cfg[0])
    return str(fmt_cfg)

def gp_plotcomplexity(
    gp,
    mode: str = "all",
    id_pop: int | None = None,
    savefig: bool | None = None,
    filename: str | None = None,
    fmt: str | None = None,
    live: bool | None = None,
):
    """
    Scatter plot: complexity vs fitness.
    - Single-pop run: plots the single population (no ensemble layer).
    - Multi-pop run: plots each population in different colors + ensemble as a unique color.
    - mode:
        * "all": all populations (and ensemble if num_pop > 1)
        * "population": requires id_pop (0-based)
    Saving:
      If savefig=True, optional 'filename' and 'fmt' (e.g., "png").
      Defaults to gp.config['runcontrol']['plot'] settings.

    Returns: absolute path to saved file (if saved), else None.
    """

    # Validate inputs & pick config defaults
    plotcfg = _safe_plot_cfg(gp)
    if savefig is None:
        savefig = plotcfg["savefig"]
    if live is None:
        live = plotcfg["live"]

    # Pull arrays prepared during evalfitness
    # isolated: shape [pop_size, num_pop]
    fit_iso = gp.individuals.get("fitness", {}).get("isolated", {}).get("train", None)
    cplx_iso = gp.individuals.get("complexity", {}).get("isolated", None)

    if fit_iso is None or cplx_iso is None:
        gp.error("Complexity/Fitness (isolated) not available. Run evolve() first.", RuntimeError)

    # Ensemble arrays exist only for multi-pop runs
    num_pop = int(gp.config["runcontrol"]["num_pop"])
    fit_en = None
    cplx_en = None
    if num_pop > 1:
        fit_en = gp.individuals.get("fitness", {}).get("ensemble", {}).get("train", None)
        cplx_en = gp.individuals.get("complexity", {}).get("ensemble", None)

    # Figure title bits
    run_name = gp.userdata.get("name", "GenForge")
    minim = bool(gp.config["runcontrol"].get("minimisation", True))
    # title_opt = r"$\mathrm{Complexity\ vs.\ Fitness}$"
    # subtitle_opt = r"$\mathrm{(lower\ is\ better)}$" if minim else r"$\mathrm{(higher\ is\ better)}$"

    # Prepare plot
    _latexish()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = _ensure_palette(num_pop + (1 if (num_pop > 1 and fit_en is not None) else 0))
    
    sns.set_theme(
        style="darkgrid",
        rc={
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        },
    )
    # Decide what to plot
    mode = (mode or "all").lower().strip()
    if mode not in ("all", "population"):
        gp.warning(f"Unknown mode='{mode}', falling back to 'all'.")

    if mode == "population":
        # Require id_pop
        if id_pop is None or not (0 <= int(id_pop) < num_pop):
            gp.error(f"'population' mode requires a valid id_pop in [0..{num_pop-1}].", ValueError)
        x = np.asarray(cplx_iso[:, int(id_pop)])
        y = np.asarray(fit_iso[:, int(id_pop)])
        ax.scatter(x, y, s=42, alpha=0.85, color=colors[int(id_pop)], label=f"Population {int(id_pop)+1}")
        # title = f"{run_name}: Complexity vs Fitness (Population {int(id_pop)+1})"
    else:
        # "all": plot each population
        for p in range(num_pop):
            x = np.asarray(cplx_iso[:, p])
            y = np.asarray(fit_iso[:, p])
            ax.scatter(x, y, s=36, alpha=0.85, color=colors[p], label=f"Population {p+1}")

        # title = f"{run_name}: Complexity vs Fitness (All Populations)"
        # and overlay ensemble if multi-pop is active and arrays exist
        if num_pop > 1 and (cplx_en is not None) and (fit_en is not None):
            x = np.asarray(cplx_en).ravel()
            y = np.asarray(fit_en).ravel()
            ax.scatter(x, y, s=52, marker="D", alpha=0.95, color=colors[-1], label="Ensemble")

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)    
    # Labels & cosmetics
    # ax.set_title(rf"{title_opt}  {subtitle_opt}", pad=8)
    ax.set_xlabel(r"$\mathrm{Complexity}$")
    ax.set_ylabel(r"$\mathrm{Fitness}$")
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.3)
    ax.legend(loc="best", frameon=True, fancybox=True)

    # Save?
    out_path = None
    if savefig:
        # pick folder (default to CWD if None)
        folder = plotcfg["folder"] or os.getcwd()
        os.makedirs(folder, exist_ok=True)

        chosen_fmt = _pick_fmt(fmt if fmt is not None else plotcfg["format"])
        # default filename if not provided
        if filename is None or not str(filename).strip():
            # mode suffix
            if mode == "population":
                base = f"{run_name}_complexity_pop{int(id_pop)}"
            else:
                base = f"{run_name}_complexity_all"
            filename = f"{base}.{chosen_fmt}"
        # If filename provided without extension & fmt given, append
        if "." not in os.path.basename(filename):
            filename = f"{filename}.{chosen_fmt}"

        out_path = os.path.abspath(os.path.join(folder, filename))
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if live:
        plt.show()
    else:
        plt.close(fig)

    return out_path
