######  install matlab engine for python --- before running example
#cd MATLABROOT/extern/engines
#python setup.py build --build-base /tmp/matbuild install --user
#####

# Some systems crash when importing matlab *after* numper, that's why we do it here */
import matlab.engine
import matlab
import numpy as np
import surfstat_wrap as sw

sw.matlab_init_surfstat()

result_py, result_mat = sw.matlab_SurfStatLinMod(5.0, 5.0)

print("yooo! ", result_py)

contrast = np.ones(1)
results = sw.matlab_SurfStatT(result_mat, contrast)

pvals = sw.matlab_SurfStatP(results)

print('pvalue: ', pvals)

asurf = np.array([[1,2,1], [1,1,2]])

aedge = sw.matlab_SurfStatEdg(asurf)

print('edges: ', aedge)
