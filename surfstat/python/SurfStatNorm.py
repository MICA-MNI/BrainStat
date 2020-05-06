import sys
import numpy as np

def py_SurfStatNorm(Y, mask=None, subdiv='s'):

	# Normalizes by subtracting the global mean, or dividing it. 
    # Inputs     	
    # Y      = numpy array of shape (n x v) or (n x v x k). 
    #          v=#vertices.
    # mask   = numpy boolean array of shape (1 x v). 
    #          True=inside the mask, False=outside. 
    # subdiv = 's' for Y=Y-Yav or 'd' for Y=Y/Yav.
    # Outputs
    # Y      = normalized data, numpy array of shape (n x v) or (n x v x k)
    # Yav    = mean of input Y along the mask, numpy array of shape (n x 1) or (n x k)   

    Y = np.array(Y, dtype='float64')

    if np.ndim(Y) < 2:
        sys.exit('input array should be np.ndims >= 2, tip: reshape it!')
    elif np.ndim(Y) == 2:
        n, v = np.shape(Y)	
        k = 1
    elif np.ndim(Y) > 2:
        n, v, k   = np.shape(Y)	

    if mask is None:
        mask = np.array(np.ones(v), dtype=bool)    

    if np.ndim(Y) == 2:
        Yav = np.mean(Y[:,mask], axis=1)
        Yav = Yav.reshape(len(Yav), 1)
    elif np.ndim(Y) > 2:
        Yav = np.mean(Y[:,mask,:], axis=1)
        Yav = np.expand_dims(Yav, axis=1)

    if subdiv == 's':
        Y = Y - Yav
    elif subdiv == 'd':
        Y = Y / Yav
    
    return Y, np.squeeze(Yav)

