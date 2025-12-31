from genforge.gpregressor import gpregressor
from genforge.gpregressor import RegressorConfig
import pandas as pd
from multiprocessing import cpu_count
import numpy as np

df = pd.read_csv("SFRC_Groups.csv", header=0)
findex = [df[col].tolist() for col in df.columns]

xtr = pd.read_csv("SFRC_Normalized_x_train.csv", header=0).to_numpy()
ytr = pd.read_csv("SFRC_Normalized_y_train.csv", header=0).to_numpy().flatten()

xval = pd.read_csv("SFRC_Normalized_x_validation.csv", header=0).to_numpy()
yval = pd.read_csv("SFRC_Normalized_y_validation.csv", header=0).to_numpy().flatten()

xts = pd.read_csv("SFRC_Normalized_x_test.csv", header=0).to_numpy()
yts = pd.read_csv("SFRC_Normalized_y_test.csv", header=0).to_numpy().flatten()

if __name__ == "__main__":
    cfg = RegressorConfig()
    # cfg.runcontrol.num_pop = 3
    # cfg.runcontrol.pop_size = 10
    # cfg.runcontrol.generations = 5
    # cfg.runcontrol.batch_job = 2
    # cfg.runcontrol.stallgen = 20
    # cfg.runcontrol.adaptgen = 15
    # cfg.runcontrol.adaptinject = True
    # cfg.runcontrol.verbose = 1
    # cfg.runcontrol.savefreq = 0
    # cfg.runcontrol.quiet = False
    # cfg.runcontrol.useparallel = False
    # cfg.runcontrol.n_jobs = 1
    # cfg.runcontrol.usecache = True
    # cfg.runcontrol.minimisation = True
    # cfg.runcontrol.tolfit = 1e-9
    # cfg.runcontrol.plotfitness = True
    # cfg.runcontrol.plotrankall = True
    # cfg.runcontrol.plotrankbest = True
    # cfg.runcontrol.plotformat = "png"
    # cfg.runcontrol.plotfolder = None
    # cfg.runcontrol.plotsavefig = True
    # cfg.runcontrol.plotlive = False
    # cfg.runcontrol.plotbackend = "auto"
    # cfg.runcontrol.track_individuals = False
    # cfg.runcontrol.resultfolder = None
    # cfg.runcontrol.random_state = 32
    
    # cfg.selection.tournament_size = 2  #[2,2]
    # cfg.selection.elite_fraction = 0.05 #[0.05, 0,05]
    # cfg.selection.elite_fraction_ensemble = 0.05 #[0.05,0.05]
    # cfg.selection.tournament_lex_pressure = True #[True, True]
    # cfg.selection.tournament_p_pareto = 0.0 #[0, 0]
    # cfg.selection.p_ensemble = 0.0    # [0,0]
    
    # cfg.nodesconst.about = "Ephemeral random constants"      # constant nodes generation method
    # cfg.nodesconst.p_ERC = 0.1 #[0.1,0.1]
    # cfg.nodesconst.p_int = 0.0 #[0,0]
    # cfg.nodesconst.range = [-10, 10] # [[-10,10],[-10,10]]
    # cfg.nodesconst.num_dec_places = 4   #[4,4]
    
    # cfg.nodesfunctions.name = ('times','minus','plus','divide')    #  [("times", "minus", "plus"),("times", "minus", "plus")]      
    # cfg.nodesfunctions.function = None    
    # cfg.nodesfunctions.arity = None            
    # cfg.nodesfunctions.active = None           
    
    # cfg.operator.p_mutate = 0.14  #[0.14,0.14]
    # cfg.operator.p_cross = 0.84  #[0.84,0.84]
    # cfg.operator.p_direct = 0.02 #[0.02,0.02]
    # cfg.operator.mutate_par = [0.9, 0.05, 0.05, 0.0, 0.0, 0.0]  #[[0.9, 0.05, 0.05, 0.0, 0.0, 0.0],[0.9, 0.05, 0.05, 0.0, 0.0, 0.0]]
    # cfg.operator.mutate_gaussian_std = 0.1  #[0.1,0.1]
    
    # cfg.gene.p_cross_hi = 0.2        #[0.2,0.2]
    # cfg.gene.hi_cross_rate = 0.5    #[0.5,0.5]
    # cfg.gene.multigene = True     #[True,True]
    # cfg.gene.max_genes = 5      #[5,5]
    
    # cfg.tree.build_method = 3      #[3,3]
    # cfg.tree.max_nodes = np.Inf       #[np.Inf,np.Inf]
    # cfg.tree.max_depth = 4    #[4,4]
    # cfg.tree.max_mutate_depth = 4  #[4,4]

    # cfg.fitness.terminate = False      
    # cfg.fitness.complexityMeasure = 1   
    
    # cfg.linregression.l1_ratio = [1.0] #[1.0,1.0]
    # cfg.linregression.alphas = None  #[None, None]
    # cfg.linregression.n_alphas = 100 #[100,100]
    # cfg.linregression.eps = 1e-3       #[None, None]
    # cfg.linregression.fit_intercept = True #[True,True]
    # cfg.linregression.copy_x = True #[True,True]
    # cfg.linregression.max_iter= 1000 #[1000,1000]
    # cfg.linregression.tol = 1e-4   #[1e-4,1e-4]
    # cfg.linregression.cv = 5 #[5, 5]
    # cfg.linregression.n_jobs = None #[None, None]
    # cfg.linregression.verbose = 0 #[0, 0]
    # cfg.linregression.positive = False       #[False,False]
    # cfg.linregression.selection = 'cyclic'       #['cyclic','cyclic']
    

    # cfg.user.name = "Example_SFRC" 
    # cfg.user.stats = True
    # cfg.user.user_fcn = None
    # cfg.user.xtrain = xtr
    # cfg.user.ytrain = ytr
    # cfg.user.xval = xval
    # cfg.user.yval = yval
    # cfg.user.xtest = xts
    # cfg.user.ytest = yts
    # cfg.user.numClass = None
    # cfg.user.pop_idx = findex   
    # cfg.user.initial_population = None 
    
    # minimal inputs
    cfg.runcontrol.num_pop = 3
    cfg.runcontrol.useparallel = False
    cfg.runcontrol.pop_size = 10
    cfg.runcontrol.generations = 150
    cfg.runcontrol.stallgen = 30
    cfg.runcontrol.random_state = 32
    cfg.runcontrol.plotsavefig = True
    cfg.runcontrol.plotfitness = True
    cfg.runcontrol.plotrankall = False
    cfg.runcontrol.plotrankbest = False
    cfg.runcontrol.plotformat = "png"
    
    cfg.user.name = "Example_SFRC" 
    cfg.user.xtrain = xtr
    cfg.user.ytrain = ytr
    cfg.user.xval = xval
    cfg.user.yval = yval
    cfg.user.xtest = xts
    cfg.user.ytest = yts
    cfg.user.pop_idx = findex   
    
    
    ens_idx = 1
    gp = gpregressor.initialize(cfg)
    gp.evolve()
    gp.report(ensemble_row=ens_idx)
    
    yptr1 = gp.predict(xtr, mode='ensemble', ensemble_row=ens_idx)
    yptr2 = gp.individuals['yp']['ensemble']['train'][ens_idx]
    e1 = np.abs(yptr1-yptr2)
    print(f'Max error for ensemble {ens_idx} is: {np.max(e1)}')
    
    
    id_pop = 0
    id_ind = 3
    yptr3 = gp.predict(xtr, mode='isolated', id_pop=id_pop, id_ind=id_ind)
    yptr4 = gp.individuals['yp']['isolated']['train'][id_pop][id_ind]
    e2 = np.abs(yptr3-yptr4)
    print(f'Max error for isolated {id_ind} of population {id_pop} is: {np.max(e2)}')
    
    gp.plotcomplexity(mode="all", savefig=True, filename="all_scatter", fmt="png")  # auto name & fmt
    gp.plotcomplexity(mode="population", id_pop=0, savefig=True, filename="pop0_scatter", fmt="png")
    
    
    
    
    