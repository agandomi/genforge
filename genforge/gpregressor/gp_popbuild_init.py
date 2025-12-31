# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
from .gp_treegen import gp_treegen
from .gp_getnumnodes import gp_getnumnodes
import copy

def gp_popbuild_init(gp):
    """Generate an initial population of GP individuals."""
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    max_nodes = gp.config['tree']['max_nodes']
    max_genes = gp.config['gene']['max_genes']
    use_multigene = gp.config['gene']['multigene']
    gen = gp.state['generation']
    initial_pop = gp.userdata['initial_population']
    rng = gp.config['runcontrol']['RNG']

    # Initialize vars
    gp.population = [[None] * pop_size for _ in range(num_pop)]

    # Building process
    for id_pop in range(num_pop):
        for i in range(pop_size):
            if initial_pop is not None and len(initial_pop[id_pop]) >= i:
                
                gp.population[id_pop][i] = copy.deepcopy(initial_pop[id_pop][i])
            else:
            
                # Override any gene settings if using single gene gp
                if not use_multigene[id_pop]:
                    max_genes = [1] * num_pop
                    
                # Loop through population and randomly pick number of genes in individual
                if max_genes[id_pop] > 1:
                    num_genes = rng.integers(1, max_genes[id_pop] + 1)
                else:
                    num_genes = 1
                
                # Construct empty individual
                individ = [None for _ in range(num_genes)]  
                # Loop through genes in each individual and generate a tree for each
                for z in range(num_genes):  
                    geneLoopCounter = 0
                    while True:
                        geneLoopCounter += 1
                        # Generate a trial tree for gene z
                        temp = gp_treegen(gp, id_pop)
                        numnodes = gp_getnumnodes(temp)
    
                        if numnodes <= max_nodes[id_pop]:
                            copyDetected = False
                            if z > 1:  # Check previous genes for copies
                                for k in range(z):
                                    if temp == individ[k]:
                                        copyDetected = True
                                        break
                        # Break the while loop if no genes are copied
                        if not copyDetected:
                            break
    
                        # Display a warning if having difficulty building trees due to constraints
                        if not gp.config['runcontrol'].get('quiet', True) and geneLoopCounter > 10:
                            print('initbuild: Reiterating tree build loop to ensure uniqueness constraints are met.')
                    
                    # Put the created gene into the individual
                    individ[z] = temp
                
                # Write new individual to population list
                gp.population[id_pop][i] = copy.deepcopy(individ)
            
    if gp.config['runcontrol']['track_individuals']:
        gp.track['population'][gen] = copy.deepcopy(gp.population)
    # return gp
