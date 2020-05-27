import matlab.engine
import matlab
from numbers import Number
import numpy as np
import numpy.matlib
import sys
sys.path.append("../surfstat")
import surfstat_wrap as sw
from term import Term

def py_SurfStatLinMod(Y, M, surf=None, niter=1, thetalim=0.01, drlim=0.1):
	""" Fits linear mixed effects models to surface data and estimates resels.
    Parameters
    ----------
    Y : numpy array of shape (n,v) or (n,v,k),
        surface data, v=#vertices, n=#observations, k=#variates.
    M : nummpy array of shape (n,p), or, a scalar, or,
        model formula of class Term or Random,
        design matrix of p regressors for the linear model.
    surf : a dictionary with key 'tri' or 'lat',
        surf['tri'] = numpy array of shape (t,3), triangle indices, or
        surf['lat'] = numpy array of shape (nx,ny,nz), 1=in, 0=out,
        (nx,ny,nz) = size(volume).
    niter : number of extra iterations of the Fisher scoring algorithm
        for fitting mixed effects models.
    thetalim : lower limit on variance coefficients, in sd's.
    drlim : step of ratio of variance coefficients, in sd's. 

    Returns
    -------
    slm : a dictionary 
        slm['X'] : numpy array of shape (n,p), design matrix.
        slm['V'] : numpy array of shape (n,n,q), variance matrix bases, 
                   normalised so that mean of the diagonal of slm['V'] is 1,
                   absent if q=1 and slm['V']=numpy.eye(n). 
        slm['df'] : int, degrees of freedom = n - rank(X)
        slm['coef'] : numpy array of shape (p,v,k), 
                   coefficients of the linear model.
        slm['SSE'] : numpy array of shape (k*(k+1)/2,v), 
                   sum of squares of errors whitened by slm['V']*slm['r']
        slm['r'] : numpy array of shape ((q-1),v), coefficients of the first 
                   (q-1) components of slm['V'] divided by their sum. 
                   Coefficients are clamped to a minimum of 0.01 x sd.
        slm['dr'] : numpy array of shape ((q-1),nc), 
                   increments in slm['r'] = 0.1 x sd.
        If surf is not empty, returns also:
        slm['resl'] : numpy array of shape (e,k), sum over observations of squa-
                   res of differences of normalized residuals along each edge.
        slm['tri'] : surf['tri'],
        slm['lat'] : surf['lat']

	"""
	if isinstance(M, Term):
		M = M.matrix.values

	maxchunk=2^20

	if isinstance(Y, Number):
		isnum = True
		Y = np.array([[Y]])
		s = np.shape(Y)
	elif isinstance(Y, np.ndarray):
		isnum = True
		if len(np.shape(Y)) == 1:
			Y = Y.reshape(1, len(Y))
		s = np.shape(Y)

	n = s[0]
	v = s[1]
	if len(s) == 2:
		k = 1
	else:
		k = s[2]

	keys = ['X', 'df', 'coef', 'SSE']
	slm = {key: None for key in keys}

	if isinstance(M, Number):
		M = np.array([[M]])
		slm['X'] = np.matlib.repmat(M,n,1)
		pinvX = np.linalg.pinv(slm['X'])
		r = np.ones((n,1)) - np.dot(pinvX*np.ones((n,1)), slm['X'])
	elif isinstance(M, np.ndarray):
		if len(np.shape(M)) == 1:
			slm['X'] = np.matlib.repmat(M.reshape(1,len(M)),n,1)
			pinvX = np.linalg.pinv(slm['X'])
			r = np.ones((n,1)) - np.dot(slm['X'], np.dot(pinvX, np.ones((n,1))))
		else:
			slm['X'] = M
			pinvX = np.linalg.pinv(slm['X'])
			r = np.ones((n,1)) - np.dot(slm['X'], pinvX * np.ones((n)))
	q = 1

	if np.square(r).mean() > np.spacing(1):
		print('Did you forget a constant term? :-)')

	p = np.shape(slm['X'])[1]
	slm['df'] = np.array([n - np.linalg.matrix_rank(slm['X'])])

	if k == 1:
		slm['coef'] = np.zeros((p,v))
	else:
		slm['coef'] = np.zeros((p,v,k))
	  
	k2 = k*(k+1)/2;
	slm['SSE'] = np.zeros((int(k2),v))

	if isnum:
		nc = 1
		chunk = v

	for ic in range(0, nc):
		ic += 1
		v1 = 1+(ic-1)*chunk
		v2 = min(v1+chunk-1,v)
		vc = v2-v1+1
		if k == 1:
			if q == 1:
				# fixed effects
				if not 'V' in slm:
					if s == (1, 1) and np.shape(M)[0]>1:
						coef = pinvX * Y[0][0]
					else:
						coef = np.dot(pinvX, Y)
					Y = Y - np.dot(slm['X'], coef)
				SSE = np.sum(np.power(Y, 2), axis=0)
			slm['coef'][:,(v1-1):(v2)] = coef
			slm['SSE'][:,(v1-1):v2] = SSE
		else:
			# multivariate
			print('multivariate ...')
			if not 'V' in slm:
				X = slm['X'];
			coef = np.zeros((p,vc,k))
			
			for j in range(0,k):
				coef[:,:,j] = np.dot(pinvX, Y[:,:,j])
				Y[:,:,j] = Y[:,:,j] - np.dot(X, coef[:,:,j])
			k2 = k * (k+1)/2;
			SSE = np.zeros((int(k2), vc))
			j = 0
			
			for j1 in range(0, k):
					for j2 in range(0, j1+1):
						SSE[j,:] = np.sum(np.multiply(Y[:,:,j1], Y[:,:,j2]), axis=0)
						j += 1
			slm['coef'][:,(v1-1):(v2),:] = coef
			slm['SSE'][:,(v1-1):v2] = SSE

	if surf is not None:
		print('surf is given... ')
		sw.matlab_init_surfstat()
		edg = sw.matlab_SurfStatEdg(surf)
		e = np.shape(edg)[0]
		e1 = edg[:,0].astype(int) -1 
		e2 = edg[:,1].astype(int) -1
		slm['tri']  = surf
		slm['resl'] = np.zeros((e, k))
		
		for j in range(1, k+1):
			jj = int(j * (j+1)/2 -1)
			normr = np.sqrt(slm['SSE'][jj,:])
			s = 0
			
			for i in range(0, n):
				if k == 1:
					u = np.divide(Y[i,:], normr)
				else:
					u = np.divide(Y[i,:,j-1], normr)
				s = s + np.square(u[e1] - u[e2])
			slm['resl'][:, j-1] = s;
	return slm




