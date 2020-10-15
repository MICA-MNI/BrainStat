import sys
sys.path.append("python")
from SurfStatAvVol import *
import surfstat_wrap as sw
import numpy as np
import pytest

sw.matlab_init_surfstat()

def dummy_test(filenames, fun = np.add, Nan = None, dimensionality = None):

    # wrap matlab functions
    M_data, M_vol = sw.matlab_SurfStatAvVol(filenames, fun, Nan, dimensionality)
   
    # run python equivalent
    P_data, P_vol = py_SurfStatAvVol(filenames, fun, Nan)
   
    if filenames[0].endswith('.img'):
        P_data = P_data[:,:,:,0]

    # compare matlab-python outputs
    testout_SurfStatAvVol = []

    for key in M_vol:
        testout_SurfStatAvVol.append(np.allclose(np.squeeze(M_vol[key]), 
                                                 np.squeeze(P_vol[key]), 
                                                 rtol=1e-05, equal_nan=True))
    testout_SurfStatAvVol.append(np.allclose(M_data, P_data, 
                                 rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout_SurfStatAvVol)
 


def test_01():
    # ANALYZE format (*img)
    filenames = ['./tests/data/volfiles/Arandom1.img',
                 './tests/data/volfiles/Arandom2.img',
                 './tests/data/volfiles/Arandom3.img',
                 './tests/data/volfiles/Arandom4.img',
                 './tests/data/volfiles/Arandom5.img']
    dummy_test(filenames)

