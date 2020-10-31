import numpy as np
import sys


def py_SurfStatStand(Y, mask=None, subtractordivide='s'):

	# Standardizes by subtracting the global mean, or dividing it.
 	# Inputs
	# Y      = numpy array of shape (n x v), v=#vertices.
	#        = NEED TO BE DISCUSSED: it works for (n x v x k) now, DO WE NEED THAT?
	# mask   = numpy boolean array of shape (1 x v).
    #          True=inside the mask, False=outside.
	# subdiv = 's' for Y=Y-Ymean or 'd' for Y=(Y/Ymean -1)*100.
	# Outputs
	# Y      = standardized data, numpy array of shape (n x v).
	# Ym     = mean of input Y along the mask, numpy array of shape (n x 1).

	Y = np.array(Y, dtype='float64')

	if mask is None:
		mask = np.array(np.ones(Y.shape[1]), dtype=bool)

	if np.ndim(Y) < 2:
		sys.exit('input array should be np.ndims >= 2, tip: reshape it!')
	elif np.ndim(Y) == 2:
		Ym = Y[:,mask].mean(axis=1)
		Ym = Ym.reshape(len(Ym), 1)
		for i in range(0, Y.shape[0]):
			if subtractordivide == 's':
				Y[i,:] = Y[i,:] - Ym[i]
			elif subtractordivide == 'd':
				Y[i,:] = (Y[i,:]/Ym[i] - 1 ) * 100;

	elif np.ndim(Y) > 2:
		Ym = np.mean(Y[:,mask,0], axis=1)
		Ym = Ym.reshape(len(Ym), 1)
		for i in range(0, Y.shape[0]):
			if subtractordivide == 's':
				Y[i,:,:] =  Y[i,:,:] - Ym[i]
			elif subtractordivide == 'd':
				Y[i,:,:] = (Y[i,:,:]/Ym[i] - 1 ) * 100;

	return Y, Ym
