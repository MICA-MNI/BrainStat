import sys
sys.path.append("python")
from SurfStatAvVol import *
import surfstat_wrap as sw
import numpy as np
import random
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
    dummy_test(np.array(filenames))

    
def test_02():
    # ANALYZE format (*img)
    filenames = ['./tests/data/volfiles/Arandom1.img',
                 './tests/data/volfiles/Arandom2.img',
                 './tests/data/volfiles/Arandom3.img',
                 './tests/data/volfiles/Arandom4.img',
                 './tests/data/volfiles/Arandom5.img']
    dummy_test(np.array(filenames), fun=np.fmin)

    
def test_03():
    # ANALYZE format (*img)
    filenames = ['./tests/data/volfiles/Arandom1.img',
                 './tests/data/volfiles/Arandom2.img',
                 './tests/data/volfiles/Arandom3.img',
                 './tests/data/volfiles/Arandom4.img',
                 './tests/data/volfiles/Arandom5.img']
    dummy_test(np.array(filenames), fun=np.fmax)


def test_04():
    # ANALYZE format (*img), image with NaN values
    filenames = ['./tests/data/volfiles/Arandom1.img',
                 './tests/data/volfiles/Arandom2.img',
                 './tests/data/volfiles/Arandom3.img',
                 './tests/data/volfiles/Arandom4.img',
                 './tests/data/volfiles/Arandom5.img',
                 './tests/data/volfiles/ArandomNaN.img']
    dummy_test(np.array(filenames), fun=np.add)
    
def test_05():
    # ANALYZE format (*img), image with NaN values, replace NaN
    filenames = ['./tests/data/volfiles/Arandom1.img',
                 './tests/data/volfiles/Arandom2.img',
                 './tests/data/volfiles/Arandom3.img',
                 './tests/data/volfiles/Arandom4.img',
                 './tests/data/volfiles/Arandom5.img',
                 './tests/data/volfiles/ArandomNaN.img']
    dummy_test(np.array(filenames), fun=np.add, Nan = random.uniform(0, 50))

def test_06():
    # NIFTI files
    filenames = ['./tests/data/volfiles/random1.nii',
                './tests/data/volfiles/random2.nii',
                './tests/data/volfiles/random3.nii',
                './tests/data/volfiles/random4.nii',
                './tests/data/volfiles/random5.nii']
    dummy_test(np.array(filenames))

def test_07():
    # NIFTI files
    filenames = ['./tests/data/volfiles/random1.nii',
                './tests/data/volfiles/random2.nii',
                './tests/data/volfiles/random3.nii',
                './tests/data/volfiles/random4.nii',
                './tests/data/volfiles/random5.nii']
    dummy_test(np.array(filenames), fun=np.fmin)

def test_08():
    # NIFTI files
    filenames = ['./tests/data/volfiles/random1.nii',
                './tests/data/volfiles/random2.nii',
                './tests/data/volfiles/random3.nii',
                './tests/data/volfiles/random4.nii',
                './tests/data/volfiles/random5.nii']
    dummy_test(np.array(filenames), fun=np.fmax)

def test_09():
    # NIFTI files
    filenames = ['./tests/data/volfiles/random1.nii',
                './tests/data/volfiles/random2.nii',
                './tests/data/volfiles/random3.nii',
                './tests/data/volfiles/random4.nii',
                './tests/data/volfiles/random5.nii',
                './tests/data/volfiles/randomNaN.nii']
    dummy_test(np.array(filenames))

def test_10():
    # NIFTI files
    filenames = ['./tests/data/volfiles/random1.nii',
                 './tests/data/volfiles/random2.nii',
                 './tests/data/volfiles/random3.nii',
                 './tests/data/volfiles/random4.nii',
                 './tests/data/volfiles/random5.nii',
                 './tests/data/volfiles/randomNaN.nii']
    dummy_test(np.array(filenames), fun=np.fmin)

def test_11():
    # NIFTI files
    filenames = ['./tests/data/volfiles/random1.nii',
                 './tests/data/volfiles/random2.nii',
                 './tests/data/volfiles/random3.nii',
                 './tests/data/volfiles/random4.nii',
                 './tests/data/volfiles/random5.nii',
                 './tests/data/volfiles/randomNaN.nii']
    dummy_test(np.array(filenames), fun=np.add, Nan=3)
