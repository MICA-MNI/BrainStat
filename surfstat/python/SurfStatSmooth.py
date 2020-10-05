import numpy as np
from numpy_groupies import aggregate
import sys
sys.path.append("python")
from SurfStatEdg import *

def py_SurfStatSmooth(Y, surf, FWHM):
    """Smooths surface data by repeatedly averaging over edges.

    Parameters
    ----------
    Y : numpy array of shape (n,v) or (n,v,k)
        surface data, v=#vertices, n=#observations, k=#variates.
    surf : a dictionary with key 'tri' or 'lat'
        surf['tri'] = numpy array of shape (t,3), triangle indices, or
        surf['lat'] = numpy array of shape (nx,ny,nz), 1=in, 0=out,
        (nx,ny,nz) = size(volume).
    FWHM : approximate FWHM of Gaussian smoothing filter, in mesh units.

    Returns
    -------
    Y : numpy array of shape (n,v) or (n,v,k),
        smoothed data.
    """
    niter = int(np.ceil(pow(FWHM,2) / (2*np.log(2))))

    if isinstance(Y, np.ndarray):
        Y = np.array(Y, dtype='float')
        if np.ndim(Y) == 2:
            n, v = np.shape(Y) 
            k = 1
            isnum = True
            
        elif np.ndim(Y) == 3:
            n, v, k = np.shape(Y)
            isnum = True

    edg = py_SurfStatEdg(surf) + 1 
    agg_1 = aggregate(edg[:,0], 2, size=(v+1))
    agg_2 = aggregate(edg[:,1], 2, size=(v+1))
    Y1 = (agg_1 + agg_2)[1:]

    if n>1:
        print(' %i x %i surfaces to smooth, %% remaining: 100 '%(n, k))

    n10 = np.floor(n/10)

    for i in range(0, n):

        if n10 != 0 and np.remainder(i+1, n10) == 0:
            print('%s ' % str(int(100-(i+1)/n10*10)), end = '')
        
        for j in range(0, k):
            if isnum:
                if np.ndim(Y) == 2:
                    Ys = Y[i,:]

                elif np.ndim(Y) == 3:
                    Ys = Y[i,:,j]

                for itera in range(1, niter+1):
                    Yedg = Ys[edg[:,0]-1] + Ys[edg[:,1]-1];    
                    agg_tmp1 = aggregate(edg[:,0], Yedg, size=(v+1))[1:]                        
                    agg_tmp2 = aggregate(edg[:,1], Yedg, size=(v+1))[1:] 
                    Ys = (agg_tmp1 + agg_tmp2) / Y1

                if np.ndim(Y) == 2:
                    Y[i,:] = Ys
                
                elif np.ndim(Y) == 3:
                    Y[i,:,j] = Ys
    if n>1:
        print('Done')

    return Y

