#####  install matlab engine for python --- before running example
#cd MATLABROOT/extern/engines
#python setup.py build --build-base /tmp/matbuild install --user
#####

import numpy as np
import surfstat_wrap as sw 

sw.init_surfstat()

slm = sw.SurfStatLinMod(5.0, 5.0)

print("yooo! ", slm)

contrast = np.ones(1)
results = sw.SurfStatT(slm, contrast)

pvals = sw.SurfStatP(results)

print('pvalue: ', pvals)
