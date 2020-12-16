import numpy as np
from brainstat.stats.SurfStatResels import SurfStatResels
from brainstat.stats.stat_threshold import stat_threshold


def SurfStatQ(slm, mask=None):
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
        mask : numpy array of shape (v), dtype 'bool',
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
        mask = np.ones((v), dtype='bool')

    df = np.zeros((2,2))
    ndf = len(np.array([slm['df']]))
    df[0, 0:ndf] = slm['df']
    df[1, 0:2] = np.array([slm['df']])[ndf-1]

    if 'dfs' in slm:
        df[0, ndf-1] = slm['dfs'][0,mask>0].mean()

    if 'du' in slm:
        resels, reselspvert, edg = SurfStatResels(slm, mask.flatten())
    else:
        reselspvert = np.ones((v))

    varA = np.append(10, slm['t'][0, mask.astype(bool)])
    P_val = stat_threshold(df = df, p_val_peak = varA,
                           nvar = float(slm['k']), nprint = 0)[0]
    P_val = P_val[1:len(P_val)]
    nx = len(P_val)
    index = P_val.argsort()
    P_sort = P_val[index]
    r_sort = reselspvert[index]
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
    qval['Q'] = np.ones((mask.shape[0]))
    qval['Q'][mask] = np.squeeze(Q[0,:])
    qval['mask'] = mask

    return qval
