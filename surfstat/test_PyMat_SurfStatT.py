import sys
sys.path.append("python")
from SurfStatT import *
import surfstat_wrap as sw
import numpy as np
import sys

sw.matlab_init_surfstat()

def dummy_test():

	try:
		# wrap matlab functions
		
		Wrapped_SurfStatT = sw.matlab_SurfStatT(Wrapped_SurfStatLinMod, C)
	
	except:
		print >> sys.stderr, "ORIGINAL MATLAB CODE DOES NOT WORK WITH THESE INPUTS..."
		sys.exit(1)

	
	# run python functions

	#Python_SurfStatLinMod = py_SurfStatLinMod(A, B)
	
	#Python_SurfStatT      = py_SurfStatT(Python_SurfStatLinMod, C)

	#testout_SurfStatLinMod = []
	#testout_SurfStatT = []

	# compare matlab-python outputs
	#for key in Wrapped_SurfStatLinMod:
	#	testout_SurfStatLinMod.append(np.allclose(Python_SurfStatLinMod[key], \
	#								  Wrapped_SurfStatLinMod[key], \
	#				   				  rtol=1e-05, equal_nan=True))

	#for key in Wrapped_SurfStatT:
	#	testout_SurfStatT.append(np.allclose(Python_SurfStatT[key], \
	#							 Wrapped_SurfStatT[key], \
	#						     rtol=1e-05, equal_nan=True) )


	#result_SurfStatLinMod = all(flag == True for (flag) in testout_SurfStatLinMod)
	#result_SurfStatT = all(flag == True for (flag) in testout_SurfStatT)

	#return  result_SurfStatLinMod, result_SurfStatT

	return

tmp = {}
tmp['X']  = np.array([[1], [1], [1], [1]])
tmp['df'] = np.array([[3.0]])
tmp['coef'] = np.array([0.3333, 0.3333, 0.3333, 0.3333]).reshape(1,4)
tmp['SSE'] = np.array([0.6667, 0.6667, 0.6667, 0.6667])


result_SurfStatLinMod, result_SurfStatT = dummy_test()

print('TEST 1a ', result_SurfStatLinMod, result_SurfStatT)



