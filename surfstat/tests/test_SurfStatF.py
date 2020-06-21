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
def test_slm1_slm2_easy_int():

    # slm1['coef'] is 2D array of integers
    # slm1['X'] and slm2['X'] are the SAME, 2D array of integers

    n = 5
    p = 6  
    k = 2
    v = 1

    rng = np.random.default_rng()

    slm1 = {}
    slm1['X'] = rng.integers(100, size=(n,p)) 
    slm1['df'] = (n-1)
    slm1['SSE'] =  rng.integers(1,100, size=(int(k*(k+1)/2),v))
    slm1['coef'] = rng.integers(100, size=(p,v))
    
    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = n
    slm2['SSE'] = rng.integers(1,100, size=(int(k*(k+1)/2),v))
    slm2['coef'] =rng.integers(100, size=(p,v))
   
    dummy_test(slm1, slm2)


#### Test 2 
def test_slm1_slm2_middle_int():

    # slm1['coef'] is 2D array of integers
    # slm1['X'] and slm2['X'] are the SAME, 2D array of integers

    n = np.random.randint(3,100)
    p = np.random.randint(3,100)  
    k = np.random.randint(3,100)
    v = np.random.randint(3,100)

    rng = np.random.default_rng()

    slm1 = {}
    slm1['X'] = rng.integers(100, size=(n,p)) 
    slm1['df'] = (n-1)
    slm1['SSE'] =  rng.integers(1,100, size=(int(k*(k+1)/2),v))
    slm1['coef'] = rng.integers(100, size=(p,v))
    
    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = n
    slm2['SSE'] = rng.integers(1,100, size=(int(k*(k+1)/2),v))
    slm2['coef'] =rng.integers(100, size=(p,v))
   
    dummy_test(slm1, slm2)

#### Test 3 
def test_slm1_slm2_easy_random():

    # slm1['coef'] is 2D random array
    # slm1['X'] and slm2['X'] are the SAME, 2D random arrays

    n = np.random.randint(3,100)
    p = np.random.randint(3,100)  
    k = np.random.randint(3,100)
    v = np.random.randint(3,100)

    rng = np.random.default_rng()

    slm1 = {}
    slm1['X'] = np.random.rand(n,p) 
    slm1['df'] = n
    slm1['SSE'] =  np.random.rand(int(k*(k+1)/2),v) 
    slm1['coef'] = np.random.rand(p,v)
    
    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p
    slm2['SSE'] = np.random.rand(int(k*(k+1)/2),v) 
    slm2['coef'] = np.random.rand(p,v)
   
    dummy_test(slm1, slm2)

#### Test 4 
def test_slm1_slm2_coef3D_int_k3():

    # k= 3
    # slm1['coef'] is 3D array of integers
    # slm1['X'] and slm2['X'] are the SAME, 2D arrays of integers

    rng = np.random.default_rng()

    n = np.random.randint(3,100)
    p = np.random.randint(3,100)
    k = 3
    v = np.random.randint(3,100)

    slm1 = {}
    slm1['X'] = rng.integers(100, size=(n,p)) 
    slm1['df'] = p
    slm1['SSE'] = rng.integers(1,100, size=(int(k*(k+1)/2),v))
    slm1['coef'] = np.ones((p,v,k)) + 2

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p+1
    slm2['SSE'] =  rng.integers(1,100, size=(int(k*(k+1)/2),v))
    slm2['coef'] = np.ones((p,v,k))
       
    dummy_test(slm1, slm2)

#### Test 5 
def test_slm1_slm2_coef3D_int_k2():
    
    # k = 2
    # slm1['coef'] is 3D array of integers
    # slm1['X'] and slm2['X'] are the SAME, 2D arrays of integers

    rng = np.random.default_rng()
    
    n = np.random.randint(3,100)
    p = np.random.randint(3,100)
    k = 2
    v = np.random.randint(3,100)

    slm1 = {}
    slm1['X'] = rng.integers(100, size=(n,p)) 
    slm1['df'] = p+1
    slm1['SSE'] = rng.integers(1,100, size=(int(k*(k+1)/2),v))
    slm1['coef'] = np.ones((p,v,k)) + 2

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p
    slm2['SSE'] =  rng.integers(1,100, size=(int(k*(k+1)/2),v))
    slm2['coef'] = np.ones((p,v,k))

    dummy_test(slm1, slm2)

#### Test 6
def test_slm1_slm2_coef3D_int_k1():
    
    # k = 1
    # slm1['coef'] is 3D array of integers
    # slm1['X'] and slm2['X'] are the SAME, 2D arrays of integers
    
    rng = np.random.default_rng()
    
    n = np.random.randint(3,100)
    p = np.random.randint(3,100)
    k = 2
    v = np.random.randint(3,100)

    slm1 = {}
    slm1['X'] = rng.integers(100, size=(n,p)) 
    slm1['df'] = n
    slm1['SSE'] =  rng.integers(1,100, size=(int(k*(k+1)/2),v))
    slm1['coef'] = rng.integers(1,100, size=(p,v,k))

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p
    slm2['SSE'] = rng.integers(1,100, size=(int(k*(k+1)/2),v))
    slm2['coef'] = rng.integers(1,100, size=(p,v,k))

    dummy_test(slm1, slm2)

#### Test 7
def test_slm1_slm2_coef3D_random_k3():

    # k= 3
    # slm1['coef'] is 3D random array
    # slm1['X'] and slm2['X'] are the SAME, 2D random array
    
    n = np.random.randint(3,100)
    p = np.random.randint(3,100)
    k = 3
    v = np.random.randint(3,100)

    slm1 = {}
    slm1['X'] = np.random.rand(n,p)
    slm1['df'] = p
    slm1['SSE'] = np.random.rand( int(k*(k+1)/2),v)
    slm1['coef'] = np.random.rand(p,v,k)

    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p+1
    slm2['SSE'] =  np.random.rand( int(k*(k+1)/2),v)
    slm2['coef'] = np.random.rand(p,v,k)
       
    dummy_test(slm1, slm2)
    
#### Test 8 
def test_slm1_slm2_coef3D_random_k2():
    
    # k = 2
    # slm1['coef'] is 3D random array
    # slm1['X'] and slm2['X'] are the SAME, 2D random array
    
    n = np.random.randint(3,100)
    p = np.random.randint(3,100)
    k = 2
    v = np.random.randint(3,100)

    slm1 = {}
    slm1['X'] = np.random.rand(n,p)
    slm1['df'] = p+1
    slm1['SSE'] = np.random.rand( int(k*(k+1)/2),v)
    slm1['coef'] = np.random.rand(p,v,k)
    
    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p
    slm2['SSE'] =  np.random.rand( int(k*(k+1)/2),v)
    slm2['coef'] = np.random.rand(p,v,k)

    dummy_test(slm1, slm2)
    
#### Test 9
def test_slm1_slm2_coef3D_random_k1():
    
    # k = 1
    # slm1['coef'] is 3D random array
    # slm1['X'] and slm2['X'] are the SAME, 2D random array
    
    n = np.random.randint(3,100)
    p = np.random.randint(3,100)
    k = 1
    v = np.random.randint(3,100)

    slm1 = {}
    slm1['X'] = np.random.rand(n,p)
    slm1['df'] = p+1
    slm1['SSE'] = np.random.rand( int(k*(k+1)/2),v)
    slm1['coef'] = np.random.rand(p,v,k)
    
    slm2 = {}
    slm2['X'] = slm1['X']
    slm2['df'] = p
    slm2['SSE'] =  np.random.rand( int(k*(k+1)/2),v)
    slm2['coef'] = np.random.rand(p,v,k)

    dummy_test(slm1, slm2)

