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


def gp_plotrankbest(gp, live: bool = False):
    """
    Plot two rankings over generations:

    1) Pop_i ranking of the best ensemble:
       For the best-performing ensemble at generation g, take its member in each
       population p and show that member's isolated rank within population p.

    2) Ensemble ranking of the best individuals:
       Build a vector from the best isolated individual of each population at
       generation g, then find that ensemble's rank among all evaluated ensembles.

    Notes:
      - Uses non-blocking live updates when 'live=True' (figure remains open).
      - No backend is selected here; choose it once before importing this module.
      - Saves to files only if config['runcontrol']['plot']['savefig'] is True.
    """
    if not gp.config['runcontrol']['plot']['rankbest']:
        return

    scatter_size = 20
    scatter_alpha = 0.5

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

    # Initialize figure/axes once
    if 'rankbest_generation' not in gp.plot:
        gp.plot['rankbest_generation'] = {}
        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots))
        gp.plot['rankbest_generation']['fig'] = fig
        gp.plot['rankbest_generation']['axes'] = axes
        if live:
            try:
                plt.ion()
                fig.show()
            except Exception:
                pass

    fig  = gp.plot['rankbest_generation']['fig']
    axes = gp.plot['rankbest_generation']['axes']
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
    colors = sns.color_palette("bright", num_pop + 1)

    # Optional jitter per population (unused by default; keep for later)
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

        # Outputs we will plot
        rank_top_en_in_iso = np.full((gen + 1, num_pop), np.nan, dtype=float)  # 1-based rank of best-ensemble member in each pop
        rank_top_iso_in_en = np.full((gen + 1,), np.nan, dtype=float)          # 1-based rank of ensemble of best isolated IDs

        for id_gen in range(gen + 1):
            # Best-first ordering of ensembles (minimisation vs maximisation)
            sort_idx = np.argsort(fit_en[id_gen]) if gp.config['runcontrol']['minimisation'] \
                       else np.argsort(-fit_en[id_gen])
            idx_en_sort = idx_en[id_gen][sort_idx, :]   # (E_sorted, num_pop)
            best_en_id  = idx_en_sort[0, :]             # member IDs per pop for the best ensemble

            # Per-pop: rank of best-ensemble member within isolated order
            iso_best_vec = np.empty((num_pop,), dtype=int)
            for p in range(num_pop):
                order = np.asarray(rank_iso_pop[id_gen][p], dtype=int)  # IDs best→worst
                iso_best_vec[p] = int(order[0])                         # best isolated ID for pop p

                # rank of the best-ensemble member in isolated order (1-based)
                pos = np.where(order == int(best_en_id[p]))[0]
                if pos.size:
                    rank_top_en_in_iso[id_gen, p] = int(pos[0]) + 1

            # Ensemble rank of the vector of best isolated IDs (if present in evaluated ensembles)
            hits = np.where((idx_en_sort == iso_best_vec).all(axis=1))[0]
            if hits.size:
                rank_top_iso_in_en[id_gen] = int(hits[0]) + 1
            # else: keep NaN (not evaluated / not present)

        # X axis
        generations = np.arange(0, gp.state['generation'] + 1)

        # Plot per-pop ranks of the best-ensemble members inside isolated rankings
        for id_pop in range(num_pop):
            ax.scatter(
                generations,
                rank_top_en_in_iso[:, id_pop],  # + offset[id_pop]  # enable jitter if you want
                label=r"$\mathrm{Pop_{%d}\ ranking\ of\ best\ ensemble}$" % (id_pop + 1,),
                s=scatter_size,
                color=colors[id_pop],
                alpha=scatter_alpha,
                edgecolors='none'
            )

        # Plot the ranking (among ensembles) of the vector made from top isolated individuals
        ax.scatter(
            generations,
            rank_top_iso_in_en,
            label=r"$\mathrm{Ensemble\ ranking\ of\ best\ individuals}$",
            s=scatter_size,
            color=colors[-1],
            alpha=scatter_alpha,
            edgecolors='none'
        )

        # Axes, limits, labels
        ax.set_xlim((-1, num_generations + 1))
        ax.set_ylim((pop_size + 1, 0))  # reverse y-axis so rank 1 is at top
        ax.set_ylabel(r"$\mathrm{Rank}$", fontsize=12)
        ax.legend(loc="best", fontsize=8)

        # Grid and ticks
        ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(0, num_generations + 1, step=1))
        y_ticks = np.arange(1, pop_size + 2, step=1)
        ax.set_yticks(y_ticks[:-1])

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
                fig.savefig(os.path.abspath(os.path.join(gp.config['runcontrol']['plot']['folder'], f"{gp.runname}_RankbestVsGeneration.{fmt}")),
                    dpi=300, format=fmt
                )
        # do not close; let caller/finalizer decide

    # Optional: final save at termination (mirrors fitness plot behavior)
    if gp.state.get('terminate', False) and gp.config['runcontrol']['plot'].get('savefig', True):
        for fmt in gp.config['runcontrol']['plot']['format']:
            fig.savefig(os.path.abspath(os.path.join(gp.config['runcontrol']['plot']['folder'], f"{gp.runname}_RankbestVsGeneration.{fmt}")),
                dpi=300, format=fmt
            )
