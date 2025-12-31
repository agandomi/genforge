# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
#from .utils importnoise_augmented_boruta 
from .gp_selection import gp_selection
from .gp_mutate import gp_mutate
from .gp_crossover import gp_crossover
from .gp_getnumnodes import gp_getnumnodes
from .gp_getdepth import gp_getdepth
from .gp_cache_enable_regress import gp_cache_enable
from .gp_copydetect import gp_copydetect
import copy

def gp_popbuild(gp):
    """Build next population of individuals."""
    
    # Set pop dependent parameters
    gen = gp.state['generation']
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    max_nodes = gp.config['tree']['max_nodes'] # pop dependent
    max_depth = gp.config['tree']['max_depth'] # pop dependent
    max_genes = gp.config['gene']['max_genes'] # pop dependent
    p_mutate_user = gp.config['operator']['p_mutate'] # pop dependent
    p_direct_user = gp.config['operator']['p_direct'] # pop dependent
    p_cross_user = gp.config['operator']['p_cross'] # pop dependent
    p_cross_hi = gp.config['gene']['p_cross_hi'] # pop dependent
    cross_rate = gp.config['gene']['hi_cross_rate'] # pop dependent
    use_multigene = gp.config['gene']['multigene'] # pop dependent
    elite_fraction_en = gp.config['selection']['elite_fraction_ensemble'] # pop dependent
    elite_fraction = gp.config['selection']['elite_fraction'] # pop dependent
    rng = gp.config['runcontrol']['RNG']
    pall = np.array(p_mutate_user) + np.array(p_direct_user) + np.array(p_cross_user)
    p_mutate = np.array(p_mutate_user) / pall
    p_direct = np.array(p_direct_user) / pall
    pmd = p_mutate + p_direct
    max_nodes_inf = np.isinf(max_nodes)
    
    # Initialize new population
    new_pop = [[None for _ in range(pop_size)] for _ in range(num_pop)]
    
    for id_pop in range(num_pop):
        # The number of new members to be constructed after elitism is accounted for
        # if gp.config['runcontrol']['useboruta']:
        #     num2boruta = gp.config['runcontrol']['borutafraction']
        # else:
        #     num2boruta = 0
        
        # number of elite individuals from ensembles
        if num_pop > 1:
            num2elite_en = int(np.ceil(elite_fraction_en[0] * pop_size))
        else:
            num2elite_en = 0
        
        # number of elite individuals from isolated population
        num2elite = int(np.ceil(elite_fraction[id_pop] * pop_size))
        
        # numer of new individuals needed to be built through mutation, crossover, etc.
        num2build = pop_size #- num2boruta

        build_count = 0
        
        # Find the existing elite individuals and append them to the new population
        iso_index = list()
        if num2elite > 0:
            sort_index = np.argsort(gp.individuals['objective']['isolated'][:, id_pop])
            if not gp.config['runcontrol']['minimisation']:
                sort_index = sort_index[::-1]
            
            copied_individual = list()
            for g in range(num2elite):
                old_index = sort_index[g]
                if g == 0:
                    best_inds = list(np.where(gp.individuals['objective']['isolated'][:, id_pop] ==\
                                              gp.individuals['objective']['isolated'][old_index, id_pop])[0]) 
                    sorted_inds = np.argsort(gp.individuals['complexity']['isolated'][best_inds, id_pop])
                    old_index = best_inds[int(sorted_inds[0])]
                    
                
                iso_index.append(old_index)
                copied_individual.append(copy.deepcopy(gp.population[id_pop][old_index]))
                
            for g in range(len(iso_index)):
                if gp.config['runcontrol']['usecache']:
                    gp_cache_enable(gp, id_pop, build_count, iso_index[g])
                    
                new_pop[id_pop][build_count] = copy.deepcopy(copied_individual[g])
                if gp.config['runcontrol']['track_individuals']:
                    gp.track['idx_minus']['elite_isolated'][gen][build_count, id_pop] = iso_index[g]
                
                build_count += 1
        
        # Append the individuals with best ensemble fitness to the new population
        en_index = list()
        if num2elite_en > 0:
            sort_index_en = np.argsort(gp.individuals['objective']['ensemble'])
            sort_index_id_en = gp.individuals['ensemble_idx'][sort_index_en, id_pop]
            
            sort_index = list()
            for g in range(len(sort_index_id_en)):
                if sort_index_id_en[g] not in sort_index:
                    sort_index.append(int(sort_index_id_en[g]))
                    
            if not gp.config['runcontrol']['minimisation']:
                sort_index = sort_index[::-1]
            
            for g in range(len(sort_index)):
                if sort_index[g] not in iso_index:
                    en_index.append(sort_index[g])
                if len(en_index) == num2elite_en:
                    break
                
            if len(en_index) < num2elite_en:
                sort_index_iso = np.argsort(gp.individuals['objective']['isolated'][:, id_pop])
                if not gp.config['runcontrol']['minimisation']:
                    sort_index_iso = sort_index_iso[::-1]
                    
                for g in sort_index_iso:
                    if g not in en_index and g not in iso_index:
                        en_index.append(g)
                    if len(en_index) == num2elite_en:
                        break
            
            copied_individual = list()
            for g in range(num2elite_en):
                copied_individual.append(copy.deepcopy(gp.population[id_pop][en_index[g]]))
                
                if gp.config['runcontrol']['usecache']:
                    gp_cache_enable(gp, id_pop, build_count, en_index[g])

                new_pop[id_pop][build_count] = copy.deepcopy(copied_individual[g])
                if gp.config['runcontrol']['track_individuals']:
                    gp.track['idx_minus']['elite_ensemble'][gen][build_count, id_pop] = en_index[g]
                
                build_count += 1
        
        # Loop until the required number of new individuals has been built.
        while build_count < num2build:
            
            # Probabilistically select a genetic operator
            p_gen = rng.random()
            if p_gen < p_mutate[id_pop]:  # Select mutation
                event_type = 1
            elif p_gen < pmd[id_pop]:  # Direct reproduction
                event_type = 2
            else:  # Crossover
                event_type = 3

            # Mutation
            # First select an individual, then a gene, then do ordinary mutate
            if event_type == 1:
                parent_index = gp_selection(gp, id_pop)  # Pick the population index of a parent individual using selection operator
                parent = copy.deepcopy(gp.population[id_pop][parent_index])

                if use_multigene[id_pop]:  # If using multigene, extract a target gene
                    num_parent_genes = len(parent)
                    target_gene_index = rng.integers(0, num_parent_genes)
                    target_gene = parent[target_gene_index]
                else:
                    target_gene_index = 0
                    target_gene = parent[target_gene_index]

                mutate_success = False
                for _ in range(10):  # Loop until a successful mutation occurs (max loops=10)
                    mutated_gene = gp_mutate(gp, target_gene, id_pop)
                    mutated_gene_depth = gp_getdepth(mutated_gene)
                    if mutated_gene_depth <= max_depth[id_pop]:
                        if max_nodes_inf[id_pop]:
                            mutate_success = True
                            break
                        mutated_gene_nodes = gp_getnumnodes(mutated_gene)
                        if mutated_gene_nodes <= max_nodes[id_pop]:
                            mutate_success = True
                            break

                # If no success then use parent gene
                if not mutate_success:
                    mutated_gene = target_gene
                    
                # Add the mutated individual to new pop
                parent[target_gene_index] = mutated_gene
                
                if gp_copydetect(new_pop, parent, id_pop):
                    continue
                else:
                    new_pop[id_pop][build_count] = copy.deepcopy(parent)
                    if gp.config['runcontrol']['track_individuals']:
                        if mutate_success:
                            gp.track['idx_minus']['mutation'][gen][build_count, id_pop] = parent_index
                        elif not mutate_success:
                            gp.track['idx_minus']['reproduction'][gen][build_count, id_pop] = parent_index
                        
                    if not mutate_success and gp.config['runcontrol']['usecache']:
                        gp_cache_enable(gp, id_pop, build_count, parent_index)
                    # new_idx[build_count][id_pop] = gp.class_['idx'][gp.state['count']-1][parent_index][jj]
                    build_count += 1

            # Direct reproduction
            elif event_type == 2:
                parent_index = gp_selection(gp, id_pop)  # Pick a parent
                parent = copy.deepcopy(gp.population[id_pop][parent_index])
                
                if gp_copydetect(new_pop, parent, id_pop):
                    continue
                else:
                    # Copy to new population
                    new_pop[id_pop][build_count] = copy.deepcopy(parent)
                    if gp.config['runcontrol']['track_individuals']:
                        gp.track['idx_minus']['reproduction'][gen][build_count, id_pop] = parent_index
                    # new_idx[build_count][jj] = gp.class_['idx'][gp.state['count']-1][parent_index][jj]
    
                    # Store fitness etc of copied individual if cache enabled
                    if gp.config['runcontrol']['usecache']:
                        gp_cache_enable(gp, id_pop, build_count, parent_index)
                    
                    build_count += 1
            # Crossover operator
            elif event_type == 3:
                high_level_cross = False
                if use_multigene[id_pop]:
                    # Select crossover type if multigene enabled
                    if rng.random() < p_cross_hi[id_pop]:
                        high_level_cross = True

                # Select parents
                parent_index = gp_selection(gp, id_pop)
                dad = copy.deepcopy(gp.population[id_pop][parent_index])
                num_dad_genes = len(dad)
                parent_index_dad = parent_index

                parent_index = gp_selection(gp, id_pop)
                mum = copy.deepcopy(gp.population[id_pop][parent_index])
                num_mum_genes = len(mum)
                parent_index_mum = parent_index

                if high_level_cross and (num_mum_genes > 1 or num_dad_genes > 1):
                    # High level crossover
                    dad_gene_selection_inds = rng.random(num_dad_genes) < cross_rate[id_pop]
                    mum_gene_selection_inds = rng.random(num_mum_genes) < cross_rate[id_pop]

                    if not np.any(dad_gene_selection_inds):
                        dad_gene_selection_inds[rng.integers(0, num_dad_genes)] = True

                    if not np.any(mum_gene_selection_inds):
                        mum_gene_selection_inds[rng.integers(0, num_mum_genes)] = True

                    dad_selected_genes = [gene for idx, gene in enumerate(dad) if dad_gene_selection_inds[idx]]
                    mum_selected_genes = [gene for idx, gene in enumerate(mum) if mum_gene_selection_inds[idx]]

                    dad_remaining_genes = [gene for idx, gene in enumerate(dad) if not dad_gene_selection_inds[idx]]
                    mum_remaining_genes = [gene for idx, gene in enumerate(mum) if not mum_gene_selection_inds[idx]]

                    mum_offspring = mum_remaining_genes + dad_selected_genes
                    dad_offspring = dad_remaining_genes + mum_selected_genes
                    
                    new_individual = copy.deepcopy(mum_offspring[:max_genes[id_pop]])
                    
                    if gp_copydetect(new_pop, new_individual, id_pop):
                        continue
                    else:
                        # Curtail offspring longer than the max allowed number of genes
                        new_pop[id_pop][build_count] = copy.deepcopy(new_individual)
                        if gp.config['runcontrol']['track_individuals']:
                            gp.track['idx_minus']['crossover_hi'][gen][id_pop][build_count,0] = parent_index_dad
                            gp.track['idx_minus']['crossover_hi'][gen][id_pop][build_count,1] = parent_index_mum
                        # new_idx[build_count][jj] = gp.class_['idx'][gp.state['count']-1][parent_index_mum][jj]
                        build_count += 1

                    if build_count < num2build:
                        new_individual = copy.deepcopy(dad_offspring[:max_genes[id_pop]])
                        
                        if gp_copydetect(new_pop, new_individual, id_pop):
                            continue
                        else:
                            new_pop[id_pop][build_count] = copy.deepcopy(new_individual)
                            if gp.config['runcontrol']['track_individuals']:
                                gp.track['idx_minus']['crossover_hi'][gen][id_pop][build_count,0] = parent_index_dad
                                gp.track['idx_minus']['crossover_hi'][gen][id_pop][build_count,1] = parent_index_mum
                            # new_idx[build_count][jj] = gp.class_['idx'][gp.state['count']-1][parent_index_dad][jj]
                            build_count += 1
                else:
                    # Low level crossover
                    if use_multigene[id_pop]:
                        dad_target_gene_num = rng.integers(0, num_dad_genes)
                        mum_target_gene_num = rng.integers(0, num_mum_genes)
                        dad_target_gene = dad[dad_target_gene_num]
                        mum_target_gene = mum[mum_target_gene_num]
                    else:
                        dad_target_gene_num = 0
                        mum_target_gene_num = 0
                        dad_target_gene = dad[dad_target_gene_num]
                        mum_target_gene = mum[mum_target_gene_num]

                    for _ in range(10):  # Loop (max 10 times) until both children meet size constraints
                        # Produce 2 offspring
                        son, daughter = gp_crossover(mum_target_gene, dad_target_gene, gp, id_pop)
                        son_depth = gp_getdepth(son)
                        
                        crossover_success = False
                        # Check if both children meet size and depth constraints
                        if son_depth <= max_depth[id_pop]:
                            daughter_depth = gp_getdepth(daughter)
                            if daughter_depth <= max_depth[id_pop]:
                                if max_nodes_inf[id_pop]:
                                    crossover_success = True
                                    break
                                son_nodes = gp_getnumnodes(son)
                                if son_nodes <= max_nodes[id_pop]:
                                    daughter_nodes = gp_getnumnodes(daughter)
                                    if daughter_nodes <= max_nodes[id_pop]:
                                        crossover_success = True
                                        break

                    # If no success then re-insert parents
                    if not crossover_success:
                        son = dad_target_gene
                        daughter = mum_target_gene
                        
                    # Write offspring back to right gene positions in parents and write to population
                    dad[dad_target_gene_num] = son
                    
                    if gp_copydetect(new_pop, dad, id_pop):
                        continue
                    else:
                        new_pop[id_pop][build_count] = copy.deepcopy(dad)
                        if gp.config['runcontrol']['track_individuals']:
                            if crossover_success:
                                gp.track['idx_minus']['crossover'][gen][id_pop][build_count,0] = parent_index_dad
                                gp.track['idx_minus']['crossover'][gen][id_pop][build_count,1] = parent_index_mum
                            elif not crossover_success:
                                gp.track['idx_minus']['reproduction'][gen][build_count, id_pop] = parent_index_dad
                                gp.track['idx_minus']['reproduction'][gen][build_count, id_pop] = parent_index_mum
                            
                        if not crossover_success and gp.config['runcontrol']['usecache']:
                            gp_cache_enable(gp, id_pop, build_count, parent_index_dad)
                            gp_cache_enable(gp, id_pop, build_count, parent_index_mum)
                    # new_idx[build_count][jj] = gp.class_['idx'][gp.state['count']-1][parent_index_dad][jj]
                    build_count += 1

                    if build_count < num2build:
                        mum[mum_target_gene_num] = daughter
                        
                        if gp_copydetect(new_pop, mum, id_pop):
                            continue
                        else:
                            new_pop[id_pop][build_count] = copy.deepcopy(mum)
                            if gp.config['runcontrol']['track_individuals']:
                                if crossover_success:
                                    gp.track['idx_minus']['crossover'][gen][id_pop][build_count,0] = parent_index_dad
                                    gp.track['idx_minus']['crossover'][gen][id_pop][build_count,1] = parent_index_mum
                                elif not crossover_success:
                                    gp.track['idx_minus']['reproduction'][gen][build_count, id_pop] = parent_index_dad
                                    gp.track['idx_minus']['reproduction'][gen][build_count, id_pop] = parent_index_mum
                                
                            if not crossover_success and gp.config['runcontrol']['usecache']:
                                gp_cache_enable(gp, id_pop, build_count, parent_index_dad)
                                gp_cache_enable(gp, id_pop, build_count, parent_index_mum)
                        #new_idx[build_count][jj] = gp.class_['idx'][gp.state['count']-1][parent_index_mum][jj]
                        build_count += 1

        # if num2boruta > 0:
        #     sort_index = np.argsort(gp.class_['fitnessindiv'][:, jj])
        #     if not gp.fitness['minimisation']:
        #         sort_index = sort_index[::-1]

        #     new_index = num2build + num2elite + num2elite_en
        #     old_index = sort_index[0]
        #     tourId = [sort_index[0]] + list(rng.choice(sort_index[1:], num2boruta-1, replace=False))
        #     popout = noise_augmented_boruta(tourId, jj, gp)

        #     new_pop[new_index][jj][cc] = popout
        #     new_idx[new_index][jj] = gp.class_['idx'][gp.state['count']-1][old_index][jj]

    gp.population = copy.deepcopy(new_pop)
    if gp.config['runcontrol']['track_individuals']:
        gp.track['population'][gen] = copy.deepcopy(new_pop)
    #gp.class_['idx'][gp.state['count']] = new_idx
    # return gp
