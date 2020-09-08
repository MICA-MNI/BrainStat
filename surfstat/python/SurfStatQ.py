import sys
import numpy as np
from scipy.io import loadmat
sys.path.append("python")
sys.path.append("../surfstat")
import surfstat_wrap as sw
import matlab.engine

# WRAPPING FUNCTIONS NEED TO BE REMOVED LATER...
sw.matlab_init_surfstat()
eng = matlab.engine.start_matlab()
eng.addpath('matlab/')
def var2mat(var):
    # Brings the input variables to matlab format.
    if isinstance(var, np.ndarray):
        var = var.tolist()
    elif var == None:
        var = []
    if not isinstance(var,list) and not isinstance(var, np.ndarray):
        var = [var]
    return matlab.double(var)

def py_SurfStatQ(slm, mask=None):
    """Q-values for False Discovey Rate of resels.

    Parameters
    ----------
    slm : a dictionary with mandatory keys 't', 'df', and 'k'.
        slm['t'] : numpy array of shape (1,v),
            v is the number of vertices.
        slm['df'] : numpy array of shape (1,1),
            degrees of freedom.
        slm['k'] : int,
            number of variates.    
    Optional parameters:
        mask : numpy array of shape (1,v), dtype 'bool',
            by default ones(1,v).        
        slm['dfs'] : numpy array of shape (1,v),
            effective degrees of freedom.
        slm['resl'] : numpy array of shape (e,v),
            matrix of sum over observations of squares of
            differences of normalized residuals along each edge.
        slm['tri'] : numpy array of shape (t,3),
            triangle indices, 1-based, t is the number of triangles,    
        or,
        slm['lat'] : 3D numpy array of 1's and 0's (1:in, 0:out).            
            
    Returns
    -------
    qval : a dictionary with keys 'Q' and 'mask'
        qval['Q'] : numpy array of shape (1,v),
            vector of Q-values.
        qval['mask'] : copy of mask.

    """    
    l, v = np.shape(slm['t'])
    
    if mask is None:
        mask = np.ones((1,v), dtype='bool')
    
    df = np.zeros((2,2))
    ndf = len(np.array([slm['df']]))
    df[0, 0:ndf] = slm['df']
    df[1, 0:2] = slm['df'][ndf-1]
    
    if 'dfs' in slm:
        df[0, ndf-1] = slm['dfs'][mask>0].mean()

    if 'du' in slm:
        # NEED TO BE CALLED FROM PYTHON SURFSTATRESELS
        resels, reselspvert, edg = sw.matlab_SurfStatResels(slm, mask)
    else:
        reselspvert = np.ones((1,v))
    reselspvert[0, mask.flatten().astype(bool)]
    
    # NEED TO BE CALLED FROM PYTHON STAT_THRESHOLD
    varA = np.append(10, slm['t'][0, mask.flatten().astype(bool)])
    P_val = np.array(eng.stat_threshold(var2mat(0),
                                        var2mat(1),
                                        var2mat(0),
                                        var2mat(df),
                                        var2mat(varA),
                                        var2mat([]),
                                        var2mat([]),
                                        var2mat([]),
                                        var2mat(slm['k']),
                                        var2mat([]),
                                        var2mat([]),
                                        var2mat(0),
                                        nargout=1))
    P_val = P_val[0,1:P_val.shape[1]]
    nx = len(P_val)
    index = P_val.argsort()
    P_sort = P_val[index]
    r_sort = reselspvert[0, index]
    c_sort = np.cumsum(r_sort)
    P_sort = P_sort / (c_sort + (c_sort <= 0)) * (c_sort > 0) * r_sort.sum()
    m = 1
    Q_sort = np.zeros((1,nx))

    for i in np.arange(nx,0,-1):
         if P_sort[i-1] < m :
             m = P_sort[i-1]
         Q_sort[0, i-1] = m

    Q = np.zeros((1,nx))
    Q[0,index] = Q_sort
    
    qval = {}
    qval['Q'] = np.ones((1, mask.shape[1]))
    qval['Q'][mask] = Q[0,:]
    qval['mask'] = mask
    
    return qval
