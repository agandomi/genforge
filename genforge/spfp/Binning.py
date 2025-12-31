# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def bin_column(i, data, n_bins, mind, maxd):
    
    dbin_column = pd.Series(0, index=range(data.shape[0]))
    xbin = np.linspace(mind[i], maxd[i], num=(n_bins+1))
    xbin1 = np.linspace(mind[i], maxd[i], num=(n_bins+1))
    xbin1[0] = xbin1[0] - 0.01/n_bins * (maxd[i] - mind[i])
    xbin1[-1] = xbin1[-1] + 0.01/n_bins * (maxd[i] - mind[i])
    for j in range(len(xbin1)-1):
        aa = data.iloc[:,i]>xbin1[j]
        bb = data.iloc[:,i]<=xbin1[j+1]
        cc = aa & bb
        dbin_column.loc[cc] = j + 1
    return (i, xbin, dbin_column)

def bin_column_with_edge(i, data, bin_edge):
    mind = data.min()
    maxd = data.max()
    
    minx = min(bin_edge[i])
    maxx = max(bin_edge[i])
    npx = len(bin_edge[i])
    if maxd[i] > maxx:
        dis = maxd[i] - maxx
        binw = bin_edge[i][-1] - bin_edge[i][-2]
        npt = np.ceil(dis/binw)
        maxd[i] = maxx + npt*binw
        npx = int(npx + npt - 1)
    if mind[i] < minx:
        dis = minx - mind[i]
        binw = bin_edge[i][1] - bin_edge[i][0]
        npb = np.ceil(dis/binw)
        mind[i] = mind[i] - npb*binw
        npx = int(npx + npb - 1)
    xbin = np.linspace(mind[i], maxd[i], num=(npx))
    xbin1 = np.linspace(mind[i], maxd[i], num=(npx))
    xbin1[0] = xbin1[0] - 0.01/(npx-1) * (maxd[i] - mind[i])
    xbin1[-1] = xbin1[-1] + 0.01/(npx-1) * (maxd[i] - mind[i])
    
    dbin_column = pd.Series(0, index=range(data.shape[0]))
    for j in range(len(xbin1)-1):
        aa = data.iloc[:,i]>xbin1[j]
        bb = data.iloc[:,i]<=xbin1[j+1]
        cc = aa & bb
        dbin_column.loc[cc] = j + 1

    return i, xbin, dbin_column

def Binning(data, n_bins = 100, method = 'equal-distance', bin_edge = None):
    
    if method == 'equal-distance' and bin_edge == None:
        
        n_jobs = -1  # Use all cores
        mind = data.min()
        maxd = data.max()
        
        results = Parallel(n_jobs=n_jobs)(delayed(bin_column)(i, data, n_bins, mind, maxd) for i in range(data.shape[1]))
        
        # Separate xbin and dbin values from results
        xbin = {result[0]: result[1] for result in results}
        dbin_list = [result[2] for result in results]
        
        # Constructing the final DataFrame from dbin_list
        dbin = pd.concat(dbin_list, axis=1)
        
    elif method == 'equal-distance' and bin_edge != None:
        
        n_jobs = -1  # Use all cores
        results = Parallel(n_jobs=n_jobs)(delayed(bin_column_with_edge)(i, data, bin_edge) for i in range(data.shape[1]))
        
        # Extracting xbin and dbin values from results
        xbin = {result[0]: result[1] for result in results}
        dbin_list = [result[2] for result in results]
        
        # Constructing the final DataFrame from dbin_list
        dbin = pd.concat(dbin_list, axis=1)
    
    return dbin, xbin
            
    