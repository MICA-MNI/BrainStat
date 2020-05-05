import warnings
from scipy.linalg import null_space
import scipy
from scipy import linalg, matrix
import numpy as np

def null(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

def py_SurfStatT(slm, contrast):
    # T statistics for a contrast in a univariate or multivariate model.
    # Inputs
    # slm         = a dict with mandatory keys 'X', 'df', 'coef', 'SSE'
    # slm['X']    = numpy array of shape (n x p), design matrix.
    # slm['df']   = numpy array of shape (a,), dtype=float64, degrees of freedom
    # slm['coef'] = numpy array of shape (p x v) or (p x v x k)
    #             = array of coefficients of the linear model. 
    #             = if (p x v), then k is thought to be 1. 
    # slm['SSE']  = numpy array of shape (k*(k+1)/2 x v)
    #             = array of sum of squares of errors   
    
    # contrast    = numpy array of shape (n x 1)
    #             = vector of contrasts in the observations, ie.
    #             = ...
   
    [n, p] = np.shape(slm['X'])
    pinvX  = np.linalg.pinv(slm['X'])

    if len(contrast) <= p:
        c = np.concatenate((contrast, \
                            np.zeros((1, p-np.shape(contrast)[1]))), axis=1).T

        if np.square(np.dot(null_space(slm['X']).T, c)).sum()  \
                / np.square(c).sum() > np.spacing(1):
            sys.exit('Contrast is not estimable :-(')

    else:
        c = np.dot(pinvX, contrast)
        r = contrast - np.dot(slm['X'], c)

        if np.square(np.ravel(r, 'F')).sum() \
                / np.square(np.ravel(contrast, 'F')).sum() >  np.spacing(1) :
            warnings.warn('Contrast is not in the model :-( ')

    slm['c']  = c.T
    slm['df'] = slm['df'][len(slm['df'])-1 ]

    if np.ndim(slm['coef']) == 2:
        k = 1
        slm['k'] = k

        if not 'r' in slm.keys():
            # fixed effect 
            if 'V' in slm.keys():
                print('NOT YET IMPLEMENTED')
                #            % Vmh=inv(chol(slm.V)');
                #            % pinvX=pinv(Vmh*slm.X);

            Vc = np.sum(np.square(np.dot(c.T, pinvX)), axis=1)
        else:
            print('NOT YET IMPLEMENTED')
            # mixed effect

        slm['ef'] = np.dot(c.T, slm['coef'])
        slm['sd'] = np.sqrt(np.multiply(Vc, slm['SSE']) / slm['df'])
        slm['t']  = np.multiply(np.divide(slm['ef'], (slm['sd']+(slm['sd']<= 0))), \
                                slm['sd']>0)

    else:
        # multivariate
        p, v, k   = np.shape(slm['coef'])
        slm['k']  = k
        slm['ef'] = np.zeros((k,v))		
	
        for j in range(0,k):
            slm['ef'][j,:] = np.dot(c.T, slm['coef'][:,:,j])
            
        j  = np.arange(1, k+1)
        jj = (np.multiply(j, j+1)/2) - 1
        jj = jj.astype(int)

        vf =  np.divide(np.sum(np.square(np.dot(c.T, pinvX)), axis=1), slm['df'])
        slm['sd'] = np.sqrt(vf * slm['SSE'][jj,:])
 

        if k == 2:
            det = np.multiply(slm['SSE'][0,:], slm['SSE'][2,:]) - \
              np.square(slm['SSE'][1,:])

            slm['t'] = np.multiply(np.square(slm['ef'][0,:]), slm['SSE'][2,:]) + \
                   np.multiply(np.square(slm['ef'][1,:]), slm['SSE'][0,:]) - \
                   np.multiply(np.multiply(2 * slm['ef'][0,:], slm['ef'][1,:]), \
                   slm['SSE'][1,:])

        if k == 3:
            det = np.multiply(slm['SSE'][0,:], (np.multiply(slm['SSE'][2,:], \
              slm['SSE'][5,:]) - np.square(slm['SSE'][4,:]))) - \
              np.multiply(slm['SSE'][5,:], np.square(slm['SSE'][1,:])) + \
              np.multiply(slm['SSE'][3,:], (np.multiply(slm['SSE'][1,:], \
              slm['SSE'][4,:]) * 2 - np.multiply(slm['SSE'][2,:], slm['SSE'][3,:])))

            slm['t'] =  np.multiply(np.square(slm['ef'][0,:]), \
                    (np.multiply(slm['SSE'][2,:], slm['SSE'][5,:]) - \
                    np.square(slm['SSE'][4,:])))

            slm['t'] = slm['t'] + np.multiply(np.square(slm['ef'][1,:]), \
                    (np.multiply(slm['SSE'][0,:], slm['SSE'][5,:]) - \
                    np.square(slm['SSE'][3,:])))

            slm['t'] = slm['t'] + np.multiply(np.square(slm['ef'][2,:]), \
                    (np.multiply(slm['SSE'][0,:], slm['SSE'][2,:]) - \
                    np.square(slm['SSE'][1,:])))

            slm['t'] = slm['t'] + np.multiply(2*slm['ef'][0,:], \
                   np.multiply(slm['ef'][1,:], (np.multiply(slm['SSE'][3,:], \
                   slm['SSE'][4,:]) - np.multiply(slm['SSE'][1,:], slm['SSE'][5,:]))))

            slm['t'] = slm['t'] + np.multiply(2*slm['ef'][0,:], \
                   np.multiply(slm['ef'][2,:], (np.multiply(slm['SSE'][1,:], \
                   slm['SSE'][4,:]) - np.multiply(slm['SSE'][2,:], slm['SSE'][3,:]))))

            slm['t'] = slm['t'] + np.multiply(2*slm['ef'][1,:], \
                   np.multiply(slm['ef'][2,:], (np.multiply(slm['SSE'][1,:], \
                   slm['SSE'][3,:]) - np.multiply(slm['SSE'][0,:], slm['SSE'][4,:]))))

        if k > 3:
            sys.exit('Hotelling''s T for k>3 not programmed yet') 

        slm['t'] = np.multiply(np.divide(slm['t'], (det + (det <= 0))), (det > 0)) / vf
        slm['t'] = np.multiply(np.sqrt(slm['t'] + (slm['t'] <= 0)), (slm['t'] > 0))

    return slm



