import BrainStatLinMod
import surfstat_wrap as sw
import numpy as np
import sys

sw.matlab_init_surfstat()

x = np.random.randint(1,10)
y = np.random.randint(1,10)

A = np.random.rand(x,y)
B = np.random.rand(x,y)

try:
    resPyWrapper, result_tmp = sw.matlab_SurfStatLinMod(A, B)
    print(resPyWrapper)

except:
    print >> sys.stderr, "ORIGINAL MATLAB CODE DOES NOT WORK WITH THESE INPUTS..."
    sys.exit(1)

resPy = BrainStatLinMod.BrainStatLinMod(A,B)
print(resPy)

for key in resPyWrapper:
	#print(resPy[key], resPyWrapper[key])
	print(key, np.allclose(resPy[key], resPyWrapper[key],
					  rtol=1e-05, equal_nan=True))


