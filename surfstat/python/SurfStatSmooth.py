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
            print("NOT YET IMPLEMENTED")

    edg = py_SurfStatEdg(surf)
    agg_1 = aggregate(edg[:,0], 2, size=(v+1))
    agg_2 = aggregate(edg[:,1], 2, size=(v+1))
    Y1 = (agg_1 + agg_2)[1:]

    if n>1:
        print(' %i x %i surfaces to smooth, %% remaining: 100 '%(n, k))

    n10 = np.floor(n/10)

    for i in range(1, n+1):
        if np.remainder(i, n10) == 0:
            print(" ADJUST THE CASE FOR n10=0" )
            print('%s ' % str(int(100-i/n10*10)), end = '')
        
        for j in range(1, k+1):
            if isnum:
                if np.ndim(Y) == 2:
                    Ys = Y[(i-1),:]
                    for itera in range(1, niter+1):
                        Yedg = Ys[edg[:,0]-1] + Ys[edg[:,1]-1];    
                        agg_tmp1 = aggregate(edg[:,0], Yedg, size=(v+1))[1:]                        
                        agg_tmp2 = aggregate(edg[:,1], Yedg, size=(v+1))[1:] 
                        Ys = (agg_tmp1 + agg_tmp2) / Y1
                    Y[(i-1),:] = Ys

                elif np.ndim(Y) == 3:
                    print('NOT YET IMPLEMENTED')
    if n>1:
        print('Done')

    return Y

Y = np.array([[2,4,5], [7,9,10]])
surf = {}
surf['tri'] = np.array([[1,2,3]])
FWHM = 3.0 

k = py_SurfStatSmooth(Y, surf, FWHM)

print(k)


