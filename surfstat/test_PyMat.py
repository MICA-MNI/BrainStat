import sys
sys.path.append("python")
from SurfStatLinMod import *
import surfstat_wrap as sw
import numpy as np

sw.matlab_init_surfstat()

def dummy_test():

	try:
		# wrap matlab functions
		Wrapped_SurfStatLinMod, bla_tmp = sw.matlab_SurfStatLinMod(A, B)  
	
	except:
		print >> sys.stderr, "ORIGINAL MATLAB CODE DOES NOT WORK WITH THESE INPUTS..."
		sys.exit(1)

	
	# run python functions
	Python_SurfStatLinMod = py_SurfStatLinMod(A, B)
	
	# compare matlab-python outputs
	testout_SurfStatLinMod = []
		
	for key in Wrapped_SurfStatLinMod:
		testout_SurfStatLinMod.append(np.allclose(Python_SurfStatLinMod[key], \
									  Wrapped_SurfStatLinMod[key], \
					   				  rtol=1e-05, equal_nan=True))

	result_SurfStatLinMod = all(flag == True for (flag) in testout_SurfStatLinMod)

	return  result_SurfStatLinMod

#### test 1a

# 1D inputs --- row vectors

v = np.random.randint(1,100)

A = np.random.rand(1,v)
B = np.random.rand(1,v)

result_SurfStatLinMod = dummy_test()
print('TEST 1a ', result_SurfStatLinMod)


#### test 1b

# 1D inputs --- column vectors

v = np.random.randint(1,100)

A = np.random.rand(v,1)
B = np.random.rand(v,1)

result_SurfStatLinMod = dummy_test()
print('TEST 1b ', result_SurfStatLinMod)


#### test 2a

# 2D inputs --- square matrices 

n = np.random.randint(1,100)

A = np.random.rand(n,n)
B = np.random.rand(n,n)

result_SurfStatLinMod = dummy_test()
print('TEST 2a ', result_SurfStatLinMod)

#### test 2b

# 2D inputs --- rectangular matrices 

n = np.random.randint(1,100)
p = np.random.randint(1,100)
v = np.random.randint(1,100)

A = np.random.rand(n,v)
B = np.random.rand(n,p)

result_SurfStatLinMod = dummy_test()

print('TEST 2b ', result_SurfStatLinMod)

#### test 3

# 3D inputs --- A is a 3D input, B is 1D 

n = np.random.randint(1,100)
p = np.random.randint(1,100)
k = np.random.randint(1,100)
v = np.random.randint(1,100)

A = np.random.rand(n,v,k)
B = np.random.rand(n,1)

result_SurfStatLinMod = dummy_test()

print('TEST 3a ', result_SurfStatLinMod)

#### test 3

# 3D inputs --- A is a 3D input, B is 2D 

n = np.random.randint(1,100)
k = np.random.randint(1,100)
v = np.random.randint(1,100)
p = np.random.randint(1,100)


A = np.random.rand(n,v,k)
B = np.random.rand(n,p)

result_SurfStatLinMod = dummy_test()

print('TEST 3b ', result_SurfStatLinMod)








