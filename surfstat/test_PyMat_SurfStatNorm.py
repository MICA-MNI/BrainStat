import sys
sys.path.append("python")
from SurfStatNorm import *
import surfstat_wrap as sw
import numpy as np

sw.matlab_init_surfstat()

def dummy_test(Y, mask, subdiv):

	try:
		# wrap matlab functions
		Wrapped_Y, Wrapped_Yav = sw.matlab_SurfStatNorm(Y, mask, subdiv)  
	
	except:
		print >> sys.stderr, "ORIGINAL MATLAB CODE DOES NOT WORK WITH THESE INPUTS..."
		sys.exit(1)

	Python_Y, Python_Yav = py_SurfStatNorm(Y, mask, subdiv)

	# compare matlab-python outputs
	testout_SurfStatNorm = []

	testout_SurfStatNorm.append(np.allclose(Wrapped_Y, Python_Y, \
					   			rtol=1e-05, equal_nan=True))

	testout_SurfStatNorm.append(np.allclose(Wrapped_Yav, Python_Yav, \
					   			rtol=1e-05, equal_nan=True))
	result_SurfStatNorm = all(flag == True for (flag) in testout_SurfStatNorm)

	return result_SurfStatNorm



#### test 1a

# 1D inputs --- row vectors

v = np.random.randint(1,9)

a = np.arange(1,v)
a = a.reshape(1, len(a))

Y = a
mask = None
subdiv = 's'

result_SurfStatNorm = dummy_test(Y, mask, subdiv)
print('TEST 1a ', result_SurfStatNorm)



#### test 1b

# 1D inputs --- row vectors & mask

a = np.arange(1,11)
a = a.reshape(1, len(a))
Y = a
mask = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
subdiv = 's'
result_SurfStatNorm = dummy_test(Y, mask, subdiv)
print('TEST 1b ', result_SurfStatNorm)



#### test 2a

# 1D inputs --- 2D arrays & mask

a = np.arange(1,11)
a = a.reshape(1, len(a))
Y = np.concatenate((a,a), axis=0)
mask = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
subdiv = 's'
result_SurfStatNorm = dummy_test(Y, mask, subdiv)
print('TEST 2a ', result_SurfStatNorm)


#### test 3a

# 1D inputs --- 3D arrays & mask


a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
Y = np.zeros((3,4,2))
Y[:,:,0] = a
Y[:,:,1] = a
mask = np.array([1, 1, 0, 0], dtype=bool)
subdiv = 's'
result_SurfStatNorm = dummy_test(Y, mask, subdiv)
print('TEST 3a ', result_SurfStatNorm)









