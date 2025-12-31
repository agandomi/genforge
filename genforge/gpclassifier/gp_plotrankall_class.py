# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

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

def compute_offset(x, base_value=0.0375):
    sign = -1 if x % 2 == 1 else 1
    multiplier = (x // 2) + 1
    return sign * multiplier * base_value


def gp_plotrankall(gp, live: bool = False):
    """
    Plot, for every population p and every isolated rank r (1..pop_size), how well
    any ensemble can perform when it MUST include that specific (p, r)-ranked
    isolated individual.

    For generation g:
      - Let order[p] be the isolated order of IDs in pop p (best→worst).
      - For each r, take id_iso = order[p][r-1] (the r-th best isolated individual).
      - Among all ensembles evaluated at g that include id_iso at column p,
        find the best (lowest) ensemble rank. Plot that as the value at g.
      - Column 0 (x = -1) is a baseline initialized to r.

    Notes:
      - Live updates when 'live=True' (non-blocking window).
      - No backend selection here; choose it once before importing this module.
      - Saves figures iff config['runcontrol']['plot']['savefig'] is True.
    """
    if not gp.config['runcontrol']['plot']['rankall']:
        return

    num_generations = gp.config['runcontrol']['num_generations']
    num_pop        = gp.config['runcontrol']['num_pop']
    pop_size       = gp.config['runcontrol']['pop_size']
    gen            = gp.state['generation']

    # Determine number of subplots (Train / Validation / Test)
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
        if live:
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

    # Optional jitter per population (kept for readability if you enable it)
    offset = np.zeros((num_pop,))
    if num_pop < 10:
        for id_pop in range(num_pop):
            offset[id_pop] = compute_offset(id_pop)
    else:
        for id_pop in range(num_pop):
            offset[id_pop] = compute_offset(id_pop, base_value=0.75 / num_pop)

    # Draw each subplot
    for ax, title, key in zip(axes, titles, data_keys):
        ax.clear()

        # Aesthetics
        ax.set_facecolor('#D3D3D3')  # light gray
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)

        # ----- Correct ranking logic -----
        # Isolated orders: [gen][pop] -> permutation of IDs (best→worst)
        rank_iso_pop = gp.plot['data']['rank']['fitness']['isolated'][key]
        # Ensemble fitness & membership indices
        fit_en = gp.plot['data']['all_ensemble']['fitness'][key]   # [gen] -> fitness per ensemble
        idx_en = gp.plot['data']['all_ensemble']['idx']            # [gen] -> (E, num_pop) individual IDs per ensemble

        # rank_ensemble[p][r-1, t] gives the best ensemble rank at "time" t
        # t = 0 corresponds to x = -1 baseline; t = g+1 corresponds to generation g
        rank_ensemble = [None for _ in range(num_pop)]
        for id_pop in range(num_pop):
            # initialize with sentinel pop_size+1 (worse than worst displayed)
            M = np.full((pop_size, gen + 2), pop_size + 1, dtype=float)
            # baseline column: 1..pop_size at x = -1
            M[:, 0] = np.arange(1, pop_size + 1, dtype=float)

            # fill per generation
            for gidx in range(gen + 1):
                # sort ensembles by objective direction
                sort_idx = np.argsort(fit_en[gidx]) if gp.config['runcontrol']['minimisation'] \
                           else np.argsort(-fit_en[gidx])
                idx_en_sort = idx_en[gidx][sort_idx, :]  # (E_sorted, num_pop)

                order = np.asarray(rank_iso_pop[gidx][id_pop], dtype=int)  # IDs best→worst in this pop

                for r0 in range(pop_size):  # r0 = 0..pop_size-1 (0-based rank; 0 is best)
                    id_iso = int(order[r0])  # the r-th best isolated individual’s ID for this pop
                    # Find the best ensemble rank that includes this ID at column id_pop
                    rows = np.where(idx_en_sort[:, id_pop] == id_iso)[0]
                    if rows.size:
                        # 1-based rank among ensembles
                        M[r0, gidx + 1] = float(rows.min() + 1)
                    # else: leave as pop_size+1 (not found / not evaluated)

            rank_ensemble[id_pop] = M

        # X axis matches matrix columns: -1, 0, 1, ..., gen
        generations = np.arange(-1, gen + 1)

        # Plot each (population, isolated-rank) line
        color_idx = 0
        for id_pop in range(num_pop):
            for r0 in range(pop_size):
                ax.plot(
                    generations,
                    rank_ensemble[id_pop][r0, :],  # + offset[id_pop]  # enable jitter if needed
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

        # LaTeX-looking tick labels without f-strings
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

    # Live vs save-only
    fig.canvas.draw_idle()
    try:
        fig.canvas.flush_events()
    except Exception:
        pass

    if live:
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
        # do not close; let caller/finalizer decide

    # Optional: final save at termination (mirror other plots)
    if gp.state.get('terminate', False) and gp.config['runcontrol']['plot'].get('savefig', True):
        for fmt in gp.config['runcontrol']['plot']['format']:
            fig.savefig(os.path.abspath(os.path.join(gp.config['runcontrol']['plot']['folder'], f"{gp.runname}_RankallVsGeneration.{fmt}")),
                dpi=300, format=fmt
            )
