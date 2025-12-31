from genforge.gpclassifier import gpclassifier
from genforge.gpclassifier import ClassifierConfig
import pandas as pd
from multiprocessing import cpu_count
import numpy as np

df = pd.read_csv("ARWPM_Groups.csv", header=0)
findex = [df[col].tolist() for col in df.columns]

xtr = pd.read_csv("ARWPM_Normalized_x_train.csv", header=0).to_numpy()
ytr = pd.read_csv("ARWPM_Normalized_y_train.csv", header=0).to_numpy()

xval = pd.read_csv("ARWPM_Normalized_x_validation.csv", header=0).to_numpy()
yval = pd.read_csv("ARWPM_Normalized_y_validation.csv", header=0).to_numpy()

xts = pd.read_csv("ARWPM_Normalized_x_test.csv", header=0).to_numpy()
yts = pd.read_csv("ARWPM_Normalized_y_test.csv", header=0).to_numpy()

batch_size = 128

if __name__ == "__main__":
    # Example parameters
    cfg = ClassifierConfig()
    # cfg.runcontrol.num_pop = 2
    # cfg.runcontrol.pop_size = 5
    # cfg.runcontrol.generations = 4
    # cfg.runcontrol.batch_job = 2
    # cfg.runcontrol.stallgen = 20
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
    
    # cfg.nodesfunctions.name = ("times", "minus", "plus")    #  [("times", "minus", "plus"),("times", "minus", "plus")]      
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
    
    # cfg.softmax.optimizer_type = "rmsprop"   #["rmsprop" ,"rmsprop" ]
    # cfg.softmax.optimizer_param = None  #[None, None]
    # cfg.softmax.initializer = "glorot_uniform"   #["glorot_uniform","glorot_uniform"]
    # cfg.softmax.regularization = None       #[None, None]
    # cfg.softmax.regularization_rate = 0.01 #[0.01,0.01]
    # cfg.softmax.batch_size = 32 #[32,32]
    # cfg.softmax.epochs= 1000 #[1000,1000]
    # cfg.softmax.patience = 10   #[10,10]
    # cfg.softmax.random_seed = None #[None, None]
    # cfg.softmax.buffer_size = None #[None, None]
    # cfg.softmax.shuffle = True #[True, True]
    # cfg.softmax.verbose = 0       #[0,0]

    # cfg.user.name = "Example_ARWM" 
    # cfg.user.stats = True
    # cfg.user.user_fcn = None
    # cfg.user.xtrain = xtr
    # cfg.user.ytrain = ytr
    # cfg.user.xval = xval
    # cfg.user.yval = yval
    # cfg.user.xtest = xts
    # cfg.user.ytest = yts
    # cfg.user.numClass = None
    # cfg.user.pop_idx = findex[:2]     
    # cfg.user.initial_population = None     
    
    # minimal implementation
    cfg.runcontrol.pop_size = 10
    cfg.runcontrol.useparallel = False
    cfg.runcontrol.generations = 100
    cfg.runcontrol.stallgen = 30
    cfg.runcontrol.random_state = 32
    cfg.runcontrol.plotsavefig = True
    cfg.runcontrol.plotfitness = True
    cfg.runcontrol.plotrankall = False
    cfg.runcontrol.plotrankbest = False
    cfg.runcontrol.plotformat = "png"
    
    cfg.user.name = "Example_ARWM" 
    cfg.user.xtrain = xtr
    cfg.user.ytrain = ytr
    cfg.user.xval = xval
    cfg.user.yval = yval
    cfg.user.xtest = xts
    cfg.user.ytest = yts
    cfg.user.pop_idx = findex[:2] 
    
   
    ens_idx = 1
    gp = gpclassifier.initialize(cfg)
    gp.evolve()
    gp.report(ensemble_row=ens_idx)
    
    pptr2 = gp.predict(xtr, mode='ensemble', ensemble_row=ens_idx, return_proba=True)
    yptr2 = gp.predict(xtr, mode='ensemble', ensemble_row=ens_idx, return_proba=False)
    yptr1 = gp.individuals['yp']['ensemble']['train'][ens_idx]
    pptr1 = gp.individuals['prob']['ensemble']['train'][ens_idx]
    e1 = np.abs(pptr1-pptr2)
    e2 = np.abs(yptr1-yptr2)
    print(f'Max probability error for ensemble {ens_idx} is: {np.max(e1)}')
    print(f'Max class error for ensemble {ens_idx} is: {np.max(e2)}')
    
    
    id_pop = 0
    id_ind = 3
    pptr3 = gp.predict(xtr, mode='isolated', id_pop=id_pop, id_ind=id_ind, return_proba=True)
    yptr3 = gp.predict(xtr, mode='isolated', id_pop=id_pop, id_ind=id_ind, return_proba=False)
    yptr4 = gp.individuals['yp']['isolated']['train'][id_pop][id_ind]
    pptr4 = gp.individuals['prob']['isolated']['train'][id_pop][id_ind]
    e4 = np.abs(pptr1-pptr2)
    e3 = np.abs(yptr1-yptr2)
    print(f'Max probability error for isolated {id_ind} of population {id_pop} is: {np.max(e4)}')
    print(f'Max class error for isolated {id_ind} of population {id_pop} is: {np.max(e3)}')
    
    gp.plotcomplexity(mode="all", savefig=True, filename="all_scatter", fmt="png")  # auto name & fmt
    gp.plotcomplexity(mode="population", id_pop=0, savefig=True, filename="pop0_scatter", fmt="png")

    
    
    
    
    
    
