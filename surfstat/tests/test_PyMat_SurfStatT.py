import sys
sys.path.append("python")
from SurfStatT import *
import surfstat_wrap as sw
import numpy as np
import sys

sw.matlab_init_surfstat()

def dummy_test(slm, contrast):

	try:
		# wrap matlab functions
		Wrapped_slm = sw.matlab_SurfStatT(slm, contrast)
	
	except:
		print >> sys.stderr, "ORIGINAL MATLAB CODE DOES NOT WORK WITH THESE INPUTS..."
		sys.exit(1)
	
	# run python functions

	Python_slm = py_SurfStatT(slm, contrast)
	
	testout_SurfStatT = []

	# compare matlab-python outputs
	for key in Wrapped_slm:
		testout_SurfStatT.append(np.allclose(Python_slm[key], Wrapped_slm[key], \
					   		     rtol=1e-05, equal_nan=True))

	result_SurfStatT = all(flag == True for (flag) in testout_SurfStatT)

	return result_SurfStatT

#### Test 1

a = np.random.randint(1,10)

tmp = {}
tmp['X']  = np.random.rand(a,1)
tmp['df'] = np.array([[3.0]])
tmp['coef'] = np.random.rand(1,a).reshape(1,a)
tmp['SSE'] = np.random.rand(1, a)

C = np.random.rand(1).reshape(1,1)

result_SurfStatT = dummy_test(slm=tmp, contrast=C)
print('Test 1a: ', result_SurfStatT)


#### Test 2  ### square matrices

a = np.random.randint(1,10)
b = np.random.randint(1,10)

tmp = {}
tmp['X']    = np.random.rand(a,a)
tmp['df']   = np.array([[b]])
tmp['coef'] = np.random.rand(a,a)
tmp['SSE']  = np.random.rand(1, a)

C = np.random.rand(1, a)

result_SurfStatT = dummy_test(slm=tmp, contrast=C)
print('Test 1b: ', result_SurfStatT)


#### Test 3  ### 3D arrays

n = 3
p = 2
v = 2
k = 3

tmp = {}
tmp['X']    = np.random.rand(n,p)
tmp['df']   = np.array([[ 1 ]])
tmp['coef'] = np.random.rand(p, v, k)
tmp['SSE']  = np.random.rand( int(k*(k+1)/2), v)
C = np.random.rand(n,1)

result_SurfStatT = dummy_test(slm=tmp, contrast=C)
print('Test 3: ', result_SurfStatT)




