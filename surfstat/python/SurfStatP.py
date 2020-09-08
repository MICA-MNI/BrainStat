import sys
import numpy as np
from scipy.interpolate import interp1d
sys.path.append("python")
from stat_threshold import stat_threshold
from SurfStatPeakClus import py_SurfStatPeakClus
from SurfStatResels import py_SurfStatResels

def py_SurfStatP(slm, mask=None, clusthresh=0.001):
    """Corrected P-values for vertices and clusters.

    Parameters
    ----------
    slm : a dictionary with keys 't', 'df', 'k', 'resl', 'tri' (or 'lat'), 
        optional key 'dfs'.
        slm['t'] : 2D numpy array of shape (l,v).
            v is the number of vertices, slm['t'][0,:] is the test statistic, 
            rest of the rows are used to calculate cluster resels if 
            slm['k']>1. See SurfStatF for the precise definition of extra rows.
        surf['df'] : 2D numpy array of shape (1,1), dtype=int.
            Degrees of freedom.
        surf['k'] : an int.
            Number of variates.
        surf['resl'] : 2D numpy array of shape (e,k).
            Sum over observations of squares of differences of normalized 
            residuals along each edge.
        surf['tri'] : 2D numpy array of shape (3,t), dtype=int.
            Triangle indices.
        or,
        surf['lat'] : 3D numpy array of shape (nx,ny,nz), 1's and 0's.
            In fact, [nx,ny,nz] = size(volume).
        surf['dfs'] : 2D numpy array of shape (1,v), dtype=int.
            Optional effective degrees of freedom.
    mask : 2D numpy array of shape (1,v), dtype=bool.
        1=inside, 0=outside, v= number of vertices. By default: np.ones((1,v), 
        dtype=bool).
    clusthresh: a float.
        P-value threshold or statistic threshold for defining clusters.
        By default: 0.001.
    
    Returns
    -------
    pval : a dictionary with keys 'P', 'C', 'mask'.
        pval['P'] : 2D numpy array of shape (1,v).
            Corrected P-values for vertices.
        pval['C'] : 2D numpy array of shape (1,v).
            Corrected P-values for clusters.
        pval['mask'] : copy of input mask.
    peak : a dictionary with keys 't', 'vertid', 'clusid', 'P'.
        peak['t'] : 2D numpy array of shape (np,1). 
            Peaks (local maxima).
        peak['vertid'] : 2D numpy array of shape (np,1). 
            Vertex.
        peak['clusid'] : 2D numpy array of shape (np,1). 
            Cluster id numbers.
        peak['P'] : 2D numpy array of shape (np,1). 
            Corrected P-values for the peak.
    clus : a dictionary with keys 'clusid', 'nverts', 'resels', 'P.'
        clus['clusid'] : 2D numpy array of shape (nc,1). 
            Cluster id numbers
        clus['nverts'] : 2D numpy array of shape (nc,1). 
            Number of vertices in cluster.
        clus['resels'] : 2D numpy array of shape (nc,1). 
            Resels in the cluster.
        clus['P'] : 2D numpy array of shape (nc,1). 
            Corrected P-values for the cluster.
    clusid : 2D numpy array of shape (1,v). 
        Cluster id's for each vertex.

    Reference: Worsley, K.J., Andermann, M., Koulis, T., MacDonald, D.
    & Evans, A.C. (1999). Detecting changes in nonisotropic images.
    Human Brain Mapping, 8:98-101.
    """
    l, v =np.shape(slm['t'])

    if mask is None:
        mask = np.ones((1,v), dtype=bool)

    df = np.zeros((2,2))
    ndf = len(slm['df'])
    df[0, 0:ndf] = slm['df']
    df[1, 0:2] = slm['df'][ndf-1]
    
    if 'dfs' in slm.keys():
        df[0, ndf-1] = slm['dfs'][mask > 0].mean()
    
    if v == 1:
        varA = varA = np.concatenate((np.array([10]), slm['t'][0]))
        pval = {}
        pval['P'] = stat_threshold(df = df, p_val_peak = varA,
                                   nvar = float(slm['k']), nprint = 0)[0]
        pval['P'] = pval['P'][1]
        peak = []
        clus = []
        clusid = []
        # only a single p-value is returned, and function is stopped.
        return pval, peak, clus, clusid
    
    if clusthresh < 1:
        thresh = stat_threshold(df = df, p_val_peak = clusthresh,
                                nvar = float(slm['k']), nprint = 0)[0]
        thresh = float(thresh[0])
    else:
        thresh = clusthresh

    resels, reselspvert, edg = py_SurfStatResels(slm, mask.flatten())
    N = mask.sum()
    
    if np.max(slm['t'][0, mask.flatten()]) < thresh:
        pval = {}
        varA = np.concatenate((np.array([[10]]), slm['t']), axis=1)
        pval['P'] = stat_threshold(search_volume = resels, num_voxels = N,
                                   fwhm = 1, df = df,
                                   p_val_peak = varA.flatten(), 
                                   nvar = float(slm['k']), nprint = 0)[0]
        pval['P'] = pval['P'][1:v+1]
        peak = []
        clus = []
        clusid = []
    else:
        peak, clus, clusid = py_SurfStatPeakClus(slm, mask, thresh,reselspvert, edg)
        slm['t'] = slm['t'].reshape(1, slm['t'].size)
        varA = np.concatenate((np.array([[10]]), peak['t'].T , slm['t']),
                              axis=1)
        varB = np.concatenate((np.array([[10]]), clus['resels']))
        pp, clpval, _, _, _, _, = stat_threshold(search_volume = resels,
                                                 num_voxels = N, fwhm = 1,
                                                 df = df,
                                                 p_val_peak = varA.flatten(),
                                                 cluster_threshold = thresh,
                                                 p_val_extent = varB,
                                                 nvar = float(slm['k']), nprint = 0)
        lenPP = len(pp[1:len(peak['t'])+1])
        peak['P'] = pp[1:len(peak['t'])+1].reshape(lenPP, 1)
        pval = {}
        pval['P'] = pp[len(peak['t']) + np.arange(1,v+1)]
        
        if slm['k'] > 1:
            print('NOT YET IMPLEMENTED')
            sys.exit()

        clus['P'] = clpval[1:len(clpval)]
        x = np.concatenate((np.array([[0]]), clus['clusid']), axis=0)
        y = np.concatenate((np.array([[1]]), clus['P']), axis=0)
        pval['C'] = interp1d(x.flatten(),y.flatten())(clusid)
            
    tlim = stat_threshold(search_volume = resels, num_voxels = N, fwhm = 1,
                          df = df, p_val_peak = np.array([0.5, 1]),
                          nvar = float(slm['k']), nprint = 0)[0]
    tlim = tlim[1]
    pval['P'] = pval['P'] * (slm['t'][0,:] > tlim) + (slm['t'][0,:] <= tlim)
    pval['mask'] = mask

    return pval, peak, clus, clusid

