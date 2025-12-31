# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os

# Define a function to format the y-tick labels
def format_ytick(y, _):
    # Adjust tolerance for rounding
    tolerance = 1e-5
    if y == 0:
        return r'$0$'  # Specifically handle the case where y is exactly 0
    elif abs(y) < 1e-3 or abs(y) > 1e3:
        return f'${{\\mathrm{{{y:.2e}}}}}$'  # Scientific notation for very small or large numbers
    elif abs(y - round(y)) < tolerance:
        return f'${{\\mathrm{{{int(y)}}}}}$'  # Integer formatting
    elif abs(y - round(y, 1)) < tolerance:
        return f'${{\\mathrm{{{y:.1f}}}}}$'  # One decimal place if close to a single decimal
    elif abs(y - round(y, 2)) < tolerance:
        return f'${{\\mathrm{{{y:.2f}}}}}$'  # Two decimal places if close to two decimals
    elif abs(y - round(y, 3)) < tolerance:
        return f'${{\\mathrm{{{y:.3f}}}}}$'  # Three decimal places if close to three decimals
    else:
        return f'${{\\mathrm{{{y:.4f}}}}}$'  # Default to four decimal places
    
def gp_plotfitness(gp, live: bool = False):
    if gp.config['runcontrol']['plot']['fitness']:
        """
        Plot the fitness vs. generations and update the plot with each generation.
        """
        num_generations = gp.config['runcontrol']['num_generations']
        num_pop = gp.config['runcontrol']['num_pop']

        # Determine number of subplots
        num_plots = 1
        if gp.userdata['xval'] is not None:
            num_plots += 1
        if gp.userdata['xtest'] is not None:
            num_plots += 1

        # Pre-calc axis limits
        min_fitness = min(np.min(gp.state['mean_fitness']['ensemble']['train'] - gp.state['std_fitness']['ensemble']['train']),
                          np.min([np.min(gp.state['mean_fitness']['isolated']['train'][id_pop] - gp.state['std_fitness']['isolated']['train'][id_pop]) for id_pop in range(num_pop)]))
        max_fitness = max(np.max(gp.state['mean_fitness']['ensemble']['train'] + gp.state['std_fitness']['ensemble']['train']),
                          np.max([np.max(gp.state['mean_fitness']['isolated']['train'][id_pop] + gp.state['std_fitness']['isolated']['train'][id_pop]) for id_pop in range(num_pop)]))

        if gp.userdata['xval'] is not None:
            min_fitness = min(min_fitness,
                              np.min(gp.state['mean_fitness']['ensemble']['validation'] - gp.state['std_fitness']['ensemble']['validation']),
                              np.min([np.min(gp.state['mean_fitness']['isolated']['validation'][id_pop] - gp.state['std_fitness']['isolated']['validation'][id_pop]) for id_pop in range(num_pop)]))
            max_fitness = max(max_fitness,
                              np.max(gp.state['mean_fitness']['ensemble']['validation'] + gp.state['std_fitness']['ensemble']['validation']),
                              np.max([np.max(gp.state['mean_fitness']['isolated']['validation'][id_pop] + gp.state['std_fitness']['isolated']['validation'][id_pop]) for id_pop in range(num_pop)]))

        if gp.userdata['xtest'] is not None:
            min_fitness = min(min_fitness,
                              np.min(gp.state['mean_fitness']['ensemble']['test'] - gp.state['std_fitness']['ensemble']['test']),
                              np.min([np.min(gp.state['mean_fitness']['isolated']['test'][id_pop] - gp.state['std_fitness']['isolated']['test'][id_pop]) for id_pop in range(num_pop)]))
            max_fitness = max(max_fitness,
                              np.max(gp.state['mean_fitness']['ensemble']['test'] + gp.state['std_fitness']['ensemble']['test']),
                              np.max([np.max(gp.state['mean_fitness']['isolated']['test'][id_pop] + gp.state['std_fitness']['isolated']['test'][id_pop]) for id_pop in range(num_pop)]))

        min_fitness *= 0.95
        max_fitness *= 1.05

        # Initialize figure/axes once
        first_time = False
        if 'fitness_generation' not in gp.plot:
            gp.plot['fitness_generation'] = {}
            fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots))
            gp.plot['fitness_generation']['fig'] = fig
            gp.plot['fitness_generation']['axes'] = axes
            first_time = True
            if live:
                try:
                    plt.ion()
                    fig.show()
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
        colors = sns.color_palette(['#ff6347', '#4682b4', '#32cd32', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'])  # Tomato, SteelBlue, LimeGreen

        for ax, title, key in zip(axes, titles, data_keys):
            ax.set_facecolor('#D3D3D3')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.5)

            ax.clear()
            generations = np.arange(gp.state['generation'] + 1)
            ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.3)

            if num_pop > 1:
                # Ensemble
                ax.plot(generations, gp.state['best']['fitness']['ensemble'][key],
                        label=r"$\mathrm{Best\ Ensemble}$", linewidth=2, color=colors[0])
                ax.fill_between(generations,
                                gp.state['mean_fitness']['ensemble'][key] - gp.state['std_fitness']['ensemble'][key],
                                gp.state['mean_fitness']['ensemble'][key] + gp.state['std_fitness']['ensemble'][key],
                                color=colors[0], alpha=0.3,
                                label=r"$\mathrm{Mean}\pm\mathrm{Std\ Ensemble}$")

                # Pops
                for id_pop in range(num_pop):
                    ax.plot(generations, gp.state['best']['fitness']['isolated'][key][id_pop],
                            label=r"$\mathrm{Best\ Pop\ %d}$" % (id_pop + 1),
                            linewidth=2, color=colors[id_pop + 1])
                    ax.fill_between(generations,
                                    gp.state['mean_fitness']['isolated'][key][id_pop] - gp.state['std_fitness']['isolated'][key][id_pop],
                                    gp.state['mean_fitness']['isolated'][key][id_pop] + gp.state['std_fitness']['isolated'][key][id_pop],
                                    color=colors[id_pop + 1], alpha=0.3,
                                    label=r"$\mathrm{Mean}\pm\mathrm{Std\ Pop\ %d}$" % (id_pop + 1))
            else:
                ax.plot(generations, gp.state['best']['fitness']['ensemble'][key],
                        label=r"$\mathrm{Best\ Individual}$", linewidth=2, color=colors[0])
                ax.fill_between(generations,
                                gp.state['mean_fitness']['ensemble'][key] - gp.state['std_fitness']['ensemble'][key],
                                gp.state['mean_fitness']['ensemble'][key] + gp.state['std_fitness']['ensemble'][key],
                                color=colors[0], alpha=0.3,
                                label=r"$\mathrm{Mean}\pm\mathrm{Std\ Population}$")

            ax.set_xlim((0, num_generations))
            ax.set_ylim((min_fitness, max_fitness))
            ax.set_ylabel(r"$\mathrm{Fitness}$", fontsize=12)
            ax.legend(loc="best", fontsize=8)

            ylabel_position = ax.yaxis.label.get_position()
            title_length = len(title)
            title_offset = (title_length / 2.0) * 0.03
            # LaTeX title without f-string
            ax.set_title(r"$\mathrm{%s}$" % title, fontsize=12,
                         loc='right', rotation=270, x=1.05, y=ylabel_position[1] - title_offset)

            max_ticks = 10
            tick_interval = max(1, num_generations // max_ticks)
            ax.set_xticks(np.arange(0, num_generations + 1, step=tick_interval))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ytick))
            ax.tick_params(axis='both', which='major', labelsize=8)

            # Tick labels in LaTeX without f-string
            ax.set_xticklabels([r"$\mathrm{%d}$" % int(x) for x in ax.get_xticks()],
                               fontdict={'fontsize': 10})
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
            if gp.config['runcontrol']['plot']['savefig']:
                for fmt in gp.config['runcontrol']['plot']['format']:
                    fig.savefig(os.path.abspath(os.path.join(gp.config['runcontrol']['plot']['folder'], f"{gp.runname}_FitnessVsGeneration.{fmt}")),
                                dpi=300, format=fmt)
            # plt.close(fig)
        
        if gp.state['terminate'] and gp.config['runcontrol']['plot']['savefig']:
            for fmt in gp.config['runcontrol']['plot']['format']:
                fig.savefig(os.path.abspath(os.path.join(gp.config['runcontrol']['plot']['folder'], f"{gp.runname}_FitnessVsGeneration.{fmt}")),
                            dpi=300, format=fmt)
            
            