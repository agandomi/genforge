# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import copy

def gp_postprocess(gp):
    
    if gp.config['runcontrol']['adaptinject'] and gp.state['terminate']:
        
        injected_exp = gp.state['injected_expression']
        num_pop = gp.config['runcontrol']['num_pop']
        pop_size = gp.config['runcontrol']['pop_size']
        pop = gp.population
        
        for ind_pop in range(num_pop):
            for ind_ind in range(pop_size):
                indiv = pop[ind_pop][ind_ind]
                for ind_g in range(len(indiv)):
                    for ind_inj in range(len(injected_exp)):
                        indiv[ind_g] = indiv[ind_g].replace(f'z{ind_inj + 1}', injected_exp[ind_inj])
                    
                pop[ind_pop][ind_ind] = copy.deepcopy(indiv)
                
        gp.population = copy.deepcopy(pop)
    
    
        best_indiv_ens = gp.state['best']['individual']['ensemble']
        for ind_gen in range(len(best_indiv_ens)):
            for ind_pop in range(len(best_indiv_ens[ind_gen])):
                for ind_g in range(len(best_indiv_ens[ind_gen][ind_pop])):
                    for ind_inj in range(len(injected_exp)):
                        best_indiv_ens[ind_gen][ind_pop][ind_g] = \
                            best_indiv_ens[ind_gen][ind_pop][ind_g].replace(f'z{ind_inj + 1}', injected_exp[ind_inj])
        
        gp.state['best']['individual']['ensemble'] = copy.deepcopy(best_indiv_ens)
        
        best_indiv_iso = gp.state['best']['individual']['isolated']
        for ind_gen in range(len(best_indiv_iso)):
            for ind_pop in range(len(best_indiv_iso[ind_gen])):
                for ind_g in range(len(best_indiv_iso[ind_gen][ind_pop])):
                    for ind_inj in range(len(injected_exp)):
                        best_indiv_iso[ind_gen][ind_pop][ind_g] = \
                            best_indiv_iso[ind_gen][ind_pop][ind_g].replace(f'z{ind_inj + 1}', injected_exp[ind_inj])
        
        gp.state['best']['individual']['isolated'] = copy.deepcopy(best_indiv_iso)
        
        
        