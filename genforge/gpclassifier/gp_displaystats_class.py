# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi


def gp_displaystats(gp):
    """Displays run stats periodically.
    
    Args:
        gp (dict): The genetic programming structure.
    """
    # Only display info if required
    if not gp.config['runcontrol']['verbose'] or gp.config['runcontrol']['quiet'] or \
            (gp.state['generation']) % gp.config['runcontrol']['verbose'] != 0:
        return

    gen = gp.state['generation']

    print(f"Generation:         {gen}")
    print(f"Best fitness:       {gp.state['best']['fitness']['ensemble']['train'][-1]:.4f}")
    print(f"Mean fitness:       {gp.state['mean_fitness']['ensemble']['train'][-1]:.4f}")
    print(f"Best complexity:    {int(gp.state['best']['complexity']['ensemble'][-1])}")
    print(f"Stall Generation:   {int(gp.state['stallgen'])}")
    print(f"Time Elapsed:       {gp.state['TimeElapsed'][-1]:.3f} sec")

    print(' ')
