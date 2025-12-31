# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi


def gp_fitness_init(gp):
    """Initializes a run."""
    gp.fitness = {
        'values': None,
        'complexities': None,
    }