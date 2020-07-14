import numpy as np
import sys
from scipy.io import loadmat
from SurfStatEdg import py_SurfStatEdg
from matlab_functions import interp1d_mp, accum, ismember

def py_SurfStatPeakClus(slm, mask, thresh, reselspvert=None, edg=None):
    
    if edg is None:
        edg = py_SurfStatEdg(slm)

    l, v = np.shape(slm['t'])
    slm['t'][0, ~mask.flatten().astype(bool)] = slm['t'][0,:].min()
    t1 = slm['t'][0, edg[:,0]-1]
    t2 = slm['t'][0, edg[:,1]-1]
    islm = np.ones((1,v))
    islm[0, [edg[t1 < t2, 0]-1]] = 0
    islm[0, [edg[t2 < t1, 1]-1]] = 0
    lmvox = np.argwhere(islm)[:,1] + 1
    excurset = np.array(slm['t'][0,:] >= thresh, dtype=int)
    n = excurset.sum()
    
    if n < 1:
        peak = []
        clus = []
        clusid = []
        return peak, clus, clusid
        sys.exit() ### HAS TO BE IMPLEMENTED NICER...
    
    voxid = np.cumsum(excurset)
    edg = voxid[edg[np.all(excurset[edg-1],1), :]-1]
    nf = np.arange(1,n+1)

    # Find cluster id's in nf (from Numerical Recipes in C, page 346):
    for el in range(1, edg.shape[0]+1):
        j = edg[el-1, 0]
        k = edg[el-1, 1]
        while nf[j-1] != j:
            j = nf[j-1]
        while nf[k-1] != k:
            k = nf[k-1]
        if j != k:
            nf[j-1] = k
            
    for j in range(1, n+1):
         while nf[j-1] != nf[nf[j-1]-1]:
             nf[j-1] =  nf[nf[j-1]-1]
 
    vox = np.argwhere(excurset) + 1
    ivox = np.argwhere(np.in1d(vox, lmvox)) + 1  
    clmid = nf[ivox-1]
    uclmid, iclmid, jclmid = np.unique(clmid, 
                                       return_index=True, return_inverse=True)
    iclmid = iclmid +1
    jclmid = jclmid +1
    ucid = np.unique(nf)
    nclus = len(ucid)
    # implementing matlab's histc function ###
    bin_edges   = np.r_[-np.Inf, 0.5 * (ucid[:-1] + ucid[1:]), np.Inf]
    ucvol, ucvol_edges = np.histogram(nf, bin_edges)
    
    if reselspvert is None:
        reselsvox = np.ones(np.shape(vox))
    else:
        reselsvox = reselspvert[0, vox-1]
        
    # calling matlab-python version for scipy's interp1d
    nf1 = interp1d_mp(np.append(0, ucid), np.arange(0,nclus+1), nf, 
                      kind='nearest')
    
    # if k>1, find volume of cluster in added sphere
    if 'k' not in slm or slm['k'] == 1:
        ucrsl = accum(nf1.astype(int).reshape(reselsvox.shape),
                      reselsvox)
    if 'k' in slm and slm['k'] == 2:
        print('NOT YET IMPLEMENTED')
        sys.exit()
        
    if 'k' in slm and slm['k'] == 3:
        print('NOT YET IMPLEMENTED')
        sys.exit()
        
    # and their ranks (in ascending order)
    iucrls = sorted(range(len(ucrsl[1:])), key=lambda k: ucrsl[1:][k])
    rankrsl = np.zeros((1, nclus))
    rankrsl[0, iucrls] =  np.arange(nclus,0,-1)
    
    lmid = lmvox[ismember(lmvox, vox)[0]]
       
    varA = slm['t'][0, (lmid-1)]
    varB = lmid
    varC = rankrsl[0,jclmid-1]
    varALL = np.concatenate((varA.reshape(len(varA),1),
                             varB.reshape(len(varB),1),
                             varC.reshape(len(varC),1)), axis=1)
    lm = np.flipud(varALL[varALL[:,0].argsort(),])
    varNEW = np.concatenate((rankrsl.T, ucvol.reshape(len(ucvol),1),
                             ucrsl.reshape(len(ucrsl),1)[1:]) , axis=1)
    cl = varNEW[varNEW[:,0].argsort(),]
    clusid = np.zeros((1,v))
    clusid[0,(vox-1).T] = interp1d_mp(np.append(0, ucid),
                                      np.append(0, rankrsl), nf,
                                      kind='nearest')
    peak = {}
    peak['t'] = lm[:,0].reshape(len(lm[:,0]), 1)
    peak['vertid'] = lm[:,1].reshape(len(lm[:,1]), 1)
    peak['clusid'] = lm[:,2].reshape(len(lm[:,2]), 1)
    clus = {}
    clus['clusid'] = cl[:,0].reshape(len(cl[:,0]), 1)
    clus['nverts'] = cl[:,1].reshape(len(cl[:,1]), 1) 
    clus['resels'] = cl[:,2] .reshape(len(cl[:,2]), 1)
    
    return peak, clus, clusid
