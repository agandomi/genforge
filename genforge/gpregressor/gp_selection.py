# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np

def gp_selection(gp, id_pop=1):
    """Select an individual from the current population."""
    
    pop_size = gp.config['runcontrol']['pop_size']
    num_pop = gp.config['runcontrol']['num_pop']
    tournament_size = gp.config['selection']['tournament_size'][id_pop]
    p_pareto = gp.config['selection']['tournament_p_pareto'][id_pop]
    p_ensemble = gp.config['selection']['p_ensemble'][id_pop] if num_pop > 1 else 0
    minimisation = gp.config['runcontrol']['minimisation']
    lex_pressure = gp.config['selection']['tournament_lex_pressure'][id_pop]
    fitness_isolated = gp.individuals['objective']['isolated'][:, id_pop]
    fitness_ensemble = gp.individuals['objective']['ensemble']
    complexity_isolated = gp.individuals['complexity']['isolated'][:, id_pop]
    complexity_ensemble = gp.individuals['complexity']['ensemble']
    rng = gp.config['runcontrol']['RNG']
    
    # Randomly choose competitors
    competitors = list(rng.choice(pop_size , tournament_size, replace=True))
    
    # determine if using ensmeble or isolated indices
    if rng.random() < p_ensemble:
        use_ensemble = True
    else:
        use_ensemble = False
        
    # choosing the ensemble or isolated fitness and complexity
    fitness_values = fitness_ensemble[competitors] if use_ensemble else fitness_isolated[competitors]
    complexities = complexity_ensemble[competitors] if use_ensemble else complexity_isolated[competitors]
    
    if rng.random() >= p_pareto:
        # Method A: Standard fitness-based tournament
        best_fitness = np.min(fitness_values) if minimisation else np.max(fitness_values)
        
        best_indices = list(np.where(np.array(fitness_values) == best_fitness)[0])
        # print(best_indices)
        
        if len(best_indices) > 1 and lex_pressure:
            best_indices = best_indices[np.argmin(complexities[best_indices])]
        else:
            best_indices = rng.choice(best_indices)
        
        ID = competitors[best_indices]
        
    else:
        # Method B: Pareto tournament
        if minimisation:
            mo = np.column_stack(( np.array(fitness_values), np.array(complexities) ))
        else:
            mo = np.column_stack(( -1 * np.array(fitness_values), np.array(complexities) ))
        
        
        competitors = competitors[~np.isinf(mo[:, 0])]
        mo = mo[~np.isinf(mo[:, 0])]
        
        if len(mo) == 0:
            return gp_selection(gp, id_pop)
        
        rank = ndfsort_rank1(mo)
        ID = rng.choice(competitors[rank])
    
    return ID

def ndfsort_rank1(mo):
    """Perform non-dominated sorting to find rank 1 individuals."""
    is_dominated = np.zeros(len(mo), dtype=bool)
    for i in range(len(mo)):
        for j in range(len(mo)):
            if all(mo[j] <= mo[i]) and any(mo[j] < mo[i]):
                is_dominated[i] = True
                break
    rank1 = list(np.where(~is_dominated)[0])
    return rank1
