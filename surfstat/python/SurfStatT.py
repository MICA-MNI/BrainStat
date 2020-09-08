import warnings
import scipy
from scipy import linalg, matrix
from scipy.linalg import null_space
from scipy.linalg import cholesky
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
    #             = if (p x v), then k is assigned to 1 here. 
    # slm['SSE']  = numpy array of shape (k*(k+1)/2 x v)
    #             = array of sum of squares of errors   
    # slm['V']    = numpy array of shape (n x n x q), variance array bases, 
    #             = normalised so that mean(diag(slm['V']))=1. If absent, assumes q=1 
    #             and slm.V=eye(n).
    # slm['r']    = numpy array of shape ((q-1) x v), coefficients of the
    #             first (q-1) components of slm['V'] divided by their sum.
    # slm['dr']   = numpy array of shape ((q-1) x 1), increments in slm['r']     
    # contrast    = numpy array of shape (n x 1), contrasts in the observations, ie.,
    #             = slm['X']*slm['c'].T, where slm['c'] is a contrast in slm.coef, or,
    #             = numpy array of shape (1 x p), of slm.c, 
    #             padded with 0's if len(contrast)<p. 
    # Outputs
    # slm['c']    = numpy array of shape (1 x p), contrasts in coefficents of the 
    #             linear model.  
    # slm['k']    = number of variates
    # slm['ef']   = numpy array of shape (k x v), array of effects.
    # slm['sd']   = numpy array of shape (k x v), standard deviations of the effects.
    # slm['t']    = numpy array of shape (1 x v), array of T = ef/sd if k=1, or,
    #             Hotelling's T if k=2 or 3, defined as the maximum T over all linear
    #             combinations of the k variates, k>3 is not programmed yet.
    # slm['dfs']  = numpy array of shape (1 x v), effective degrees of freedom.
    #             Absent if q=1.
    
	#% Note that the contrast in the observations is used to determine the
	#% intended contrast in the model coefficients, slm.c. However there is some
	#% ambiguity in this when the model contains redundant terms. An example of
	#% such a model is 1 + Gender (Gender by itself does not contain redundant 
	#% terms). Only one of the ambiguous contrasts is estimable (i.e. has slm.sd
	#% < Inf), and this is the one chosen, though it may not be the contrast
	#% that you intended. To check this, compare the contrast in the 
	#% coefficients slm.c to the actual design matrix in slm.X. Note that the
	#% redundant columns of the design matrix have weights given by the rows of
	#% null(slm.X,'r')'
  
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
                Vmh = np.linalg.inv(cholesky(slm['V']).T)
                pinvX = np.linalg.pinv(np.dot(Vmh, slm['X']))

            Vc = np.sum(np.square(np.dot(c.T, pinvX)), axis=1)
        else:
            # mixed effect
            q1, v = np.shape(slm['r'])
            q = q1 + 1
            nc = np.shape(slm['dr'])[1]
            chunck = np.ceil(v / nc)
            irs = np.zeros((q1, v))
            
            for ic in range(1, nc+1):
            	v1 = 1 + (ic - 1) * chunck
            	v2 = np.min((v1 + chunck - 1, v))
            	vc = v2 - v1 + 1
            	
            	irs[:, int(v1-1):int(v2)] = np.around(np.multiply(\
            	 slm['r'][:, int(v1-1):int(v2)], \
            	 np.tile(1/slm['dr'][:,(ic-1)], (1,vc))))
            	
            ur, ir, jr = np.unique(irs, axis=0, return_index=True, return_inverse=True)	
            ir = ir + 1
            jr = jr + 1
            nr = np.shape(ur)[0]
            slm['dfs'] = np.zeros((1,v))
            Vc = np.zeros((1,v))
            

            for ir in range(1, nr+1):
                iv = (jr == ir).astype(int) 
                rv = slm['r'][:, (iv-1)].mean(axis=1)
                V = (1 - rv.sum()) * slm['V'][:,:,(q-1)]
                
                for j in range(1, q1+1):
                    V = V + rv[(j-1)] * slm['V'][:,:,(j-1)]	
                    
                Vinv = np.linalg.inv(V)
                VinvX = np.dot(Vinv, slm['X'])
                Vbeta = np.linalg.pinv(np.dot(slm['X'].T, VinvX))
                G = np.dot(Vbeta, VinvX.T)
                Gc = np.dot(G.T, c)
               	R = Vinv - np.dot(VinvX, G)
               	E = np.zeros((q,1))
               	RVV = np.zeros((np.shape(slm['V'])))
               	M = np.zeros((q,q))
               
                for j in range(1, q+1):
                    E[(j-1)] = np.dot(Gc.T, np.dot(slm['V'][:,:,(j-1)], Gc))
                    RVV[:,:,(j-1)] = np.dot(R, slm['V'][:,:,(j-1)])
                    
                for j1 in range(1, q+1):
                    for j2 in range(j1, q+1):
                        M[(j1-1),(j2-1)] = (RVV[:,:,(j1-1)] * RVV[:,:,(j2-1)].T).sum()
                        M[(j2-1),(j1-1)] = M[(j1-1),(j2-1)]
                
                vc = np.dot(c.T, np.dot(Vbeta, c))
                iv = (jr == ir).astype(int) 
                Vc[iv-1] = vc 
                slm['dfs'][iv-1] = np.square(vc) / np.dot(E.T, \
                 np.dot(np.linalg.pinv(M), E))
 
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

