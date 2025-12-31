# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
from collections import Counter

def gp_copydetect(new_pop, parent, id_pop):
    identify_copy = False
    parent_counter = Counter(parent)  # Count the elements in the parent list

    for id_ind in range(len(new_pop[id_pop])):
        if new_pop[id_pop][id_ind] is not None:
            new_individual_counter = Counter(new_pop[id_pop][id_ind])  # Count the elements in the new individual
            if parent_counter == new_individual_counter:  # Compare the element counts
                identify_copy = True
                break

    return identify_copy
