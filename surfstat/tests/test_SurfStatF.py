import sys
sys.path.append("python")
from SurfStatF import *
import surfstat_wrap as sw
import numpy as np
import sys

sw.matlab_init_surfstat()

def dummy_test(A, B):

	try:
		# wrap matlab functions
		Wrapped_slm = sw.matlab_SurfStatF(A, B)
		
	except:
		print >> sys.stderr, "ORIGINAL MATLAB CODE DOES NOT WORK WITH THESE INPUTS..."
		sys.exit(1)
	
	# run python functions
	Python_slm = py_SurfStatF(A, B)
	
	testout_SurfStatF = []
	# compare matlab-python outputs
	for key in Wrapped_slm:
		testout_SurfStatF.append(np.allclose(Python_slm[key], Wrapped_slm[key], \
					   		     rtol=1e-05, equal_nan=True))
	assert all(flag == True for (flag) in testout_SurfStatF)


#### Test 1 
def test_slm1_slm2_easy():

    # slm1['coef'] is 2D array of integers
    # slm1['X'] is 2D array of integers

    n = 5
    p = 6    
    k = 1
    v = 2

    rng = np.random.default_rng()

    slm1 = {}
    slm1['X'] = rng.integers(100, size=(n,p))
    slm1['df'] = (n-1)
    slm1['SSE'] =  rng.integers(100, size=(int(k*(k+1)/2),v))
    slm1['coef'] = rng.integers(100, size=(p,v))
    
    slm2 = {}
    slm2['X'] = rng.integers(100, size=(n,p))
    slm2['df'] = n
    slm2['SSE'] = rng.integers(100, size=(int(k*(k+1)/2),v))
    slm2['coef'] =rng.integers(100, size=(p,v))
   
    dummy_test(slm1, slm2)


