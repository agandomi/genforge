# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import copy
def gp_cache_enable(gp, id_pop, build_count, parent_index):
    gp.cache['gene_output']['train'][id_pop][build_count] =                     copy.deepcopy(gp.individuals['gene_output']['train'][id_pop][parent_index])    
    gp.cache['gene_output']['validation'][id_pop][build_count] =                copy.deepcopy(gp.individuals['gene_output']['validation'][id_pop][parent_index])
    gp.cache['gene_output']['test'][id_pop][build_count] =                      copy.deepcopy(gp.individuals['gene_output']['test'][id_pop][parent_index])
    gp.cache['gene_penalty']['train'][id_pop][build_count] =                    copy.deepcopy(gp.individuals['gene_penalty']['train'][id_pop][parent_index])    
    gp.cache['gene_penalty']['validation'][id_pop][build_count] =               copy.deepcopy(gp.individuals['gene_penalty']['validation'][id_pop][parent_index])
    gp.cache['gene_penalty']['test'][id_pop][build_count] =                     copy.deepcopy(gp.individuals['gene_penalty']['test'][id_pop][parent_index])
    gp.cache['loss']['isolated']['train'][id_pop][build_count] =                gp.individuals['loss']['isolated']['train'][parent_index, id_pop]
    gp.cache['loss']['isolated']['validation'][id_pop][build_count] =           gp.individuals['loss']['isolated']['validation'][parent_index, id_pop]
    gp.cache['loss']['isolated']['test'][id_pop][build_count] =                 gp.individuals['loss']['isolated']['test'][parent_index, id_pop]
    gp.cache['yp']['isolated']['train'][id_pop][build_count] =                  copy.deepcopy(gp.individuals['yp']['isolated']['train'][id_pop][parent_index])
    gp.cache['yp']['isolated']['validation'][id_pop][build_count] =             copy.deepcopy(gp.individuals['yp']['isolated']['validation'][id_pop][parent_index])
    gp.cache['yp']['isolated']['test'][id_pop][build_count] =                   copy.deepcopy(gp.individuals['yp']['isolated']['test'][id_pop][parent_index])
    gp.cache['fitness']['isolated']['train'][id_pop][build_count] =             gp.individuals['fitness']['isolated']['train'][parent_index, id_pop]
    gp.cache['fitness']['isolated']['validation'][id_pop][build_count] =        gp.individuals['fitness']['isolated']['validation'][parent_index, id_pop]
    gp.cache['fitness']['isolated']['test'][id_pop][build_count] =              gp.individuals['fitness']['isolated']['test'][parent_index, id_pop]
    gp.cache['complexity']['isolated'][id_pop][build_count] =                   gp.individuals['complexity']['isolated'][parent_index, id_pop]
    gp.cache['depth']['isolated'][id_pop][build_count] =                        copy.deepcopy(gp.individuals['depth']['isolated'][id_pop][parent_index])
    gp.cache['num_nodes']['isolated'][id_pop][build_count] =                    copy.deepcopy(gp.individuals['num_nodes']['isolated'][id_pop][parent_index])
    gp.cache['weight_genes'][id_pop][build_count] =                             copy.deepcopy(gp.individuals['weight_genes'][id_pop][parent_index])
    gp.cache['objective']['isolated'][id_pop][build_count] =                    copy.deepcopy(gp.individuals['objective']['isolated'][parent_index, id_pop])