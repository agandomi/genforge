# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os

# Optional: global style (Computer Modern)
plt.rcParams.update({
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'mathtext.rm': 'serif',
    'mathtext.it': 'serif:italic',
    'mathtext.bf': 'serif:bold',
    'font.size': 12,
})

def _is_interactive_backend() -> bool:
    """Return True if the current Matplotlib backend is a GUI backend."""
    b = mpl.get_backend().lower()
    return any(k in b for k in ('qt', 'tk', 'wx', 'gtk', 'macosx'))

def compute_offset(x, base_value=0.0375):
    sign = -1 if x % 2 == 1 else 1
    multiplier = (x // 2) + 1
    return sign * multiplier * base_value


def gp_plotrankall(gp, live: bool = False):
    """
    Regression: For each population p and each isolated rank r (1..pop_size),
    plot the best (lowest) ensemble rank achievable at each generation g by any
    ensemble that MUST include that (p, r)-ranked isolated individual.

    For generation g:
      - order[p] := isolated order (IDs best→worst) for population p
      - id_iso   := order[p][r-1]
      - Among all evaluated ensembles at g with column p == id_iso, find the
        best (lowest) ensemble rank and plot it.
      - A baseline column at x = -1 is set to r (1..pop_size).
    """
    if not gp.config['runcontrol']['plot'].get('rankall', False):
        return

    num_generations = int(gp.config['runcontrol']['num_generations'])
    num_pop        = int(gp.config['runcontrol']['num_pop'])
    pop_size       = int(gp.config['runcontrol']['pop_size'])
    gen            = int(gp.state['generation'])

    # Subplots: Train / (Validation) / (Test)
    num_plots = 1
    if gp.userdata['xval'] is not None:
        num_plots += 1
    if gp.userdata['xtest'] is not None:
        num_plots += 1

    # Initialize figure/axes once (consistent key)
    if 'rankall_generation' not in gp.plot:
        gp.plot['rankall_generation'] = {}
        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots))
        gp.plot['rankall_generation']['fig'] = fig
        gp.plot['rankall_generation']['axes'] = axes
        if live and _is_interactive_backend():
            try:
                plt.ion()
                fig.show()
            except Exception:
                pass

    fig  = gp.plot['rankall_generation']['fig']
    axes = gp.plot['rankall_generation']['axes']
    if num_plots == 1:
        axes = [axes]  # normalize to list

    # Titles / data keys
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
    colors = sns.color_palette("bright", max(1, num_pop * pop_size))

    # Optional jitter per population (kept for parity; off by default)
    offset = np.zeros((num_pop,), dtype=float)
    if num_pop < 10:
        for p in range(num_pop):
            offset[p] = compute_offset(p)
    else:
        for p in range(num_pop):
            offset[p] = compute_offset(p, base_value=0.75 / num_pop)

    # Draw each subplot
    for ax, title, key in zip(axes, titles, data_keys):
        ax.clear()

        # Aesthetics
        ax.set_facecolor('#D3D3D3')  # light gray
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)

        # ----- Ranking logic -----
        # Isolated orders: rank_iso_pop[g][p] -> permutation of IDs best→worst for population p
        rank_iso_pop = gp.plot['data']['rank']['fitness']['isolated'][key]  # list length gen+1
        # Ensemble fitness & membership indices (per generation)
        fit_en = gp.plot['data']['all_ensemble']['fitness'][key]             # list length gen+1; each is (E,) array/list
        idx_en = gp.plot['data']['all_ensemble']['idx']                      # list length gen+1; each is (E, num_pop) int IDs

        # rank_ensemble[p] -> matrix (pop_size, gen+2)
        # column 0 corresponds to x = -1 baseline (1..pop_size),
        # columns 1..gen+1 correspond to generations 0..gen.
        rank_ensemble = [None for _ in range(num_pop)]
        for p in range(num_pop):
            M = np.full((pop_size, gen + 2), float(pop_size + 1), dtype=float)
            M[:, 0] = np.arange(1, pop_size + 1, dtype=float)  # baseline
            rank_ensemble[p] = M

        for gidx in range(gen + 1):
            # Skip if missing data
            if fit_en[gidx] is None or idx_en[gidx] is None:
                continue

            fit_g = np.asarray(fit_en[gidx])
            idx_g = np.asarray(idx_en[gidx])
            if fit_g.ndim != 1 or idx_g.ndim != 2 or idx_g.shape[1] != num_pop:
                continue

            # Sort ensembles by objective direction
            if gp.config['runcontrol']['minimisation']:
                order_en = np.argsort(fit_g)
            else:
                order_en = np.argsort(-fit_g)
            idx_en_sort = idx_g[order_en, :]  # (E_sorted, num_pop)

            for p in range(num_pop):
                order_iso = np.asarray(rank_iso_pop[gidx][p], dtype=int)  # IDs best→worst for pop p
                if order_iso.ndim != 1 or order_iso.size == 0:
                    continue

                # For each isolated rank r0 (0-based), find best ensemble rank including that ID at column p
                for r0 in range(pop_size):
                    id_iso = int(order_iso[r0])
                    rows = np.where(idx_en_sort[:, p] == id_iso)[0]
                    if rows.size:
                        # 1-based ensemble rank
                        rank_ensemble[p][r0, gidx + 1] = float(rows.min() + 1)
                    # else: remains pop_size+1 (not present / not evaluated)

        # X axis values: -1, 0, 1, ..., gen
        generations = np.arange(-1, gen + 1)

        # Plot each (population, isolated-rank) line
        color_idx = 0
        for p in range(num_pop):
            for r0 in range(pop_size):
                ax.plot(
                    generations,
                    rank_ensemble[p][r0, :],  # + offset[p]  # enable jitter if desired
                    linewidth=1,
                    color=colors[color_idx],
                )
                color_idx = (color_idx + 1) % len(colors)

        # Axes, limits, labels
        ax.set_xlim((-1, num_generations + 1))
        ax.set_ylim((pop_size + 1, 0))  # reverse y-axis so rank 1 is at top
        ax.set_ylabel(r"$\mathrm{Rank}$", fontsize=12)

        # Grid and ticks
        ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(0, num_generations + 1, step=1))
        y_ticks = np.arange(1, pop_size + 1, step=1)
        ax.set_yticks(y_ticks)

        # Use ticker.FuncFormatter
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y)}'))
        ax.tick_params(axis='both', which='major', labelsize=8, labelcolor='black')

        # LaTeX-looking tick labels
        ax.set_xticklabels([r"$\mathrm{%d}$" % int(x) for x in ax.get_xticks()],
                           fontdict={'fontsize': 10})
        ax.set_yticklabels([r"$\mathrm{%d}$" % int(y) for y in ax.get_yticks()],
                           fontdict={'fontsize': 10})

        # Title on the right, vertical
        ylabel_position = ax.yaxis.label.get_position()
        title_length = len(title)
        title_offset = (title_length / 2.0) * 0.03
        ax.set_title(r"$\mathrm{%s}$" % title, fontsize=12,
                     loc='right', rotation=270, x=1.05, y=ylabel_position[1] - title_offset)

        ax.set_xlabel(r"$\mathrm{Generation}$", fontsize=12)
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    plt.tight_layout()

    # Live vs save-only (only do GUI actions if backend is interactive)
    fig.canvas.draw_idle()
    try:
        fig.canvas.flush_events()
    except Exception:
        pass

    if live and _is_interactive_backend():
        try:
            plt.pause(0.001)
        except Exception:
            pass
    else:
        if gp.config['runcontrol']['plot'].get('savefig', True):
            for fmt in gp.config['runcontrol']['plot']['format']:
                fig.savefig(os.path.abspath(os.path.join(gp.config['runcontrol']['plot']['folder'], f"{gp.runname}_RankallVsGeneration.{fmt}")),
                    dpi=300, format=fmt
                )

    # Final save at termination (no closing)
    if gp.state.get('terminate', False) and gp.config['runcontrol']['plot'].get('savefig', True):
        for fmt in gp.config['runcontrol']['plot']['format']:
            fig.savefig(os.path.abspath(os.path.join(gp.config['runcontrol']['plot']['folder'], f"{gp.runname}_RankallVsGeneration.{fmt}")),
                dpi=300, format=fmt
            )
