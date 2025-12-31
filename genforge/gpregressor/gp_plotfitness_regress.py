# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ---- helpers ----------------------------------------------------------------
def _is_interactive_backend() -> bool:
    """True if the active Matplotlib backend supports interactive windows."""
    try:
        b = matplotlib.get_backend().lower()
    except Exception:
        return False
    interactive = ("qt5agg", "qtagg", "tkagg", "wxagg", "gtk3agg", "macosx", "nbagg", "webagg")
    return any(k in b for k in interactive)

def format_ytick(y, _):
    """Pretty mathtext formatting for y-ticks."""
    tol = 1e-5
    if y == 0:
        return r"$0$"
    if abs(y) < 1e-3 or abs(y) > 1e3:
        return f"${{\\mathrm{{{y:.2e}}}}}$"
    if abs(y - round(y)) < tol:
        return f"${{\\mathrm{{{int(y)}}}}}$"
    if abs(y - round(y, 1)) < tol:
        return f"${{\\mathrm{{{y:.1f}}}}}$"
    if abs(y - round(y, 2)) < tol:
        return f"${{\\mathrm{{{y:.2f}}}}}$"
    if abs(y - round(y, 3)) < tol:
        return f"${{\\mathrm{{{y:.3f}}}}}$"
    return f"${{\\mathrm{{{y:.4f}}}}}$"

# ---- main -------------------------------------------------------------------
def gp_plotfitness(gp, live: bool = False):
    """
    Regression Fitness-vs-Generation plot (mirrors classifier look/feel).
    Respects chosen backend; only shows/pauses when interactive.
    """
    if not gp.config['runcontrol']['plot']['fitness']:
        return

    num_generations = gp.config['runcontrol']['num_generations']
    num_pop = gp.config['runcontrol']['num_pop']

    # number of subplots
    num_plots = 1
    if gp.userdata['xval'] is not None:
        num_plots += 1
    if gp.userdata['xtest'] is not None:
        num_plots += 1

    # axis limits (same logic as your classifier)
    min_fitness = min(
        np.min(gp.state['mean_fitness']['ensemble']['train'] - gp.state['std_fitness']['ensemble']['train']),
        np.min([np.min(gp.state['mean_fitness']['isolated']['train'][p]
                     - gp.state['std_fitness']['isolated']['train'][p])
                for p in range(num_pop)])
    )
    max_fitness = max(
        np.max(gp.state['mean_fitness']['ensemble']['train'] + gp.state['std_fitness']['ensemble']['train']),
        np.max([np.max(gp.state['mean_fitness']['isolated']['train'][p]
                     + gp.state['std_fitness']['isolated']['train'][p])
                for p in range(num_pop)])
    )

    if gp.userdata['xval'] is not None:
        min_fitness = min(
            min_fitness,
            np.min(gp.state['mean_fitness']['ensemble']['validation'] - gp.state['std_fitness']['ensemble']['validation']),
            np.min([np.min(gp.state['mean_fitness']['isolated']['validation'][p]
                         - gp.state['std_fitness']['isolated']['validation'][p])
                    for p in range(num_pop)])
        )
        max_fitness = max(
            max_fitness,
            np.max(gp.state['mean_fitness']['ensemble']['validation'] + gp.state['std_fitness']['ensemble']['validation']),
            np.max([np.max(gp.state['mean_fitness']['isolated']['validation'][p]
                         + gp.state['std_fitness']['isolated']['validation'][p])
                    for p in range(num_pop)])
        )

    if gp.userdata['xtest'] is not None:
        min_fitness = min(
            min_fitness,
            np.min(gp.state['mean_fitness']['ensemble']['test'] - gp.state['std_fitness']['ensemble']['test']),
            np.min([np.min(gp.state['mean_fitness']['isolated']['test'][p]
                         - gp.state['std_fitness']['isolated']['test'][p])
                    for p in range(num_pop)])
        )
        max_fitness = max(
            max_fitness,
            np.max(gp.state['mean_fitness']['ensemble']['test'] + gp.state['std_fitness']['ensemble']['test']),
            np.max([np.max(gp.state['mean_fitness']['isolated']['test'][p]
                         + gp.state['std_fitness']['isolated']['test'][p])
                    for p in range(num_pop)])
        )

    min_fitness *= 0.95
    max_fitness *= 1.05

    # init once; only show if live AND interactive
    if 'fitness_generation' not in gp.plot:
        gp.plot['fitness_generation'] = {}
        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots))
        gp.plot['fitness_generation']['fig'] = fig
        gp.plot['fitness_generation']['axes'] = axes
        if live and _is_interactive_backend():
            try:
                plt.ion()
                plt.show(block=False)
            except Exception:
                pass

    fig = gp.plot['fitness_generation']['fig']
    axes = gp.plot['fitness_generation']['axes']
    if num_plots == 1:
        axes = [axes]

    titles = ['Train']
    if gp.userdata['xval'] is not None:
        titles.append('Validation')
    if gp.userdata['xtest'] is not None:
        titles.append('Test')

    data_keys = ['train']
    if gp.userdata['xval'] is not None:
        data_keys.append('validation')
    if gp.userdata['xtest'] is not None:
        data_keys.append('test')

    sns.set(style="darkgrid")
    colors = sns.color_palette(['#ff6347', '#4682b4', '#32cd32', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'])

    for ax, title, key in zip(axes, titles, data_keys):
        ax.set_facecolor('#D3D3D3')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)

        ax.clear()
        generations = np.arange(gp.state['generation'] + 1)
        ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.3)

        if num_pop > 1:
            ax.plot(
                generations,
                gp.state['best']['fitness']['ensemble'][key],
                label=r"$\mathrm{Best\ Ensemble}$",
                linewidth=2, color=colors[0]
            )
            ax.fill_between(
                generations,
                gp.state['mean_fitness']['ensemble'][key] - gp.state['std_fitness']['ensemble'][key],
                gp.state['mean_fitness']['ensemble'][key] + gp.state['std_fitness']['ensemble'][key],
                color=colors[0], alpha=0.3,
                label=r"$\mathrm{Mean}\pm\mathrm{Std\ Ensemble}$"
            )
            for p in range(num_pop):
                ax.plot(
                    generations,
                    gp.state['best']['fitness']['isolated'][key][p],
                    label=r"$\mathrm{Best\ Pop\ %d}$" % (p + 1),
                    linewidth=2, color=colors[(p + 1) % len(colors)]
                )
                ax.fill_between(
                    generations,
                    gp.state['mean_fitness']['isolated'][key][p] - gp.state['std_fitness']['isolated'][key][p],
                    gp.state['mean_fitness']['isolated'][key][p] + gp.state['std_fitness']['isolated'][key][p],
                    color=colors[(p + 1) % len(colors)], alpha=0.3,
                    label=r"$\mathrm{Mean}\pm\mathrm{Std\ Pop\ %d}$" % (p + 1)
                )
        else:
            ax.plot(
                generations,
                gp.state['best']['fitness']['ensemble'][key],
                label=r"$\mathrm{Best\ Individual}$",
                linewidth=2, color=colors[0]
            )
            ax.fill_between(
                generations,
                gp.state['mean_fitness']['ensemble'][key] - gp.state['std_fitness']['ensemble'][key],
                gp.state['mean_fitness']['ensemble'][key] + gp.state['std_fitness']['ensemble'][key],
                color=colors[0], alpha=0.3,
                label=r"$\mathrm{Mean}\pm\mathrm{Std\ Population}$"
            )

        ax.set_xlim((0, num_generations))
        ax.set_ylim((min_fitness, max_fitness))
        ax.set_ylabel(r"$\mathrm{Fitness}$", fontsize=12)
        ax.legend(loc="best", fontsize=8)

        ylabel_pos = ax.yaxis.label.get_position()
        title_offset = (len(title) / 2.0) * 0.03
        ax.set_title(
            r"$\mathrm{%s}$" % title,
            fontsize=12, loc='right', rotation=270, x=1.05, y=ylabel_pos[1] - title_offset
        )

        max_ticks = 10
        tick_interval = max(1, num_generations // max_ticks)
        ax.set_xticks(np.arange(0, num_generations + 1, step=tick_interval))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ytick))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xticklabels([r"$\mathrm{%d}$" % int(x) for x in ax.get_xticks()],
                           fontdict={'fontsize': 10})
        ax.set_xlabel(r"$\mathrm{Generation}$", fontsize=12)
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    plt.tight_layout()

    fig.canvas.draw_idle()
    try:
        fig.canvas.flush_events()
    except Exception:
        pass

    # Only pause if interactive backend AND live plotting requested
    if live and _is_interactive_backend():
        try:
            plt.pause(0.001)
        except Exception:
            pass
    else:
        if gp.config['runcontrol']['plot']['savefig']:
            for fmt in gp.config['runcontrol']['plot']['format']:
                fig.savefig(os.path.abspath(os.path.join(gp.config['runcontrol']['plot']['folder'], f"{gp.runname}_FitnessVsGeneration.{fmt}")),
                    dpi=300, format=fmt
                )

    if gp.state['terminate'] and gp.config['runcontrol']['plot']['savefig']:
        for fmt in gp.config['runcontrol']['plot']['format']:
            fig.savefig(os.path.abspath(os.path.join(gp.config['runcontrol']['plot']['folder'], f"{gp.runname}_FitnessVsGeneration.{fmt}")),
                dpi=300, format=fmt
            )
