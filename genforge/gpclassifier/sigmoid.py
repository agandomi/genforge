# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np

def sigmoid(input):
    """Compute the sigmoid function for the input."""
    sigmoidval = 1 / (1 + np.exp(-input))
    return sigmoidval
