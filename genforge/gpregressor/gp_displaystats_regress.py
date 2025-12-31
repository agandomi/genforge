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

    # Multiple independent run count display

    print(f"Generation:             {gen}")
    print(f"Best objective:         {gp.state['best']['objective']['ensemble'][-1]:.4f}")
    print(f"Best fitness train:     {gp.state['best']['fitness']['ensemble']['train'][-1]:.4f}")
    if gp.userdata['yval'] is not None:
        print(f"Best fitness valid:     {gp.state['best']['fitness']['ensemble']['validation'][-1]:.4f}")
    if gp.userdata['ytest'] is not None:
        print(f"Best fitness test:      {gp.state['best']['fitness']['ensemble']['test'][-1]:.4f}")
    print(f"Best penalty train:     {gp.state['best']['penalty']['ensemble']['train'][-1]:.4f}")
    if gp.userdata['yval'] is not None:
        print(f"Best penalty valid:     {gp.state['best']['penalty']['ensemble']['validation'][-1]:.4f}")
    if gp.userdata['ytest'] is not None:
        print(f"Best penalty test:      {gp.state['best']['penalty']['ensemble']['test'][-1]:.4f}")
    print(f"Mean fitness:           {gp.state['mean_fitness']['ensemble']['train'][-1]:.4f}")
    print(f"Best complexity:        {int(gp.state['best']['complexity']['ensemble'][-1])}")
    print(f"Stall Generation:       {int(gp.state['stallgen'])}")
    print(f"Time Elapsed:           {gp.state['TimeElapsed'][-1]:.3f} sec")
    # # Display inputs in "best training" individual, if enabled.
    # if gp['runcontrol']['showBestInputs']:
    #     numx, hitvec = gpmodelvars(gp, 'best')
    #     inputs = ''.join([f"x{num} " for num in numx])
    #     print(f"Inputs in best individual: {inputs}")

    # # Display inputs in "best validation" individual, if enabled.
    # if gp['runcontrol']['showValBestInputs'] and 'xval' in gp['userdata'] and \
    #         gp['userdata']['xval'] and 'valbest' in gp['results']:
    #     numx, hitvec = gpmodelvars(gp, 'valbest')
    #     inputs = ''.join([f"x{num} " for num in numx])
    #     print(f"Inputs in best validation individual: {inputs}")

    print(' ')
