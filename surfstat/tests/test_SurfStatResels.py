import sys
sys.path.append("python")
from SurfStatResels import py_SurfStatResels
from SurfStatEdg import py_SurfStatEdg
import numpy as np
import matlab.engine
import math
import itertools
import pytest

eng = matlab.engine.start_matlab()
eng.addpath('matlab/')

def matlab_SurfStatResels(slm, mask=None): 
    # slm.resl = numpy array of shape (e,k)
    # slm.tri  = numpy array of shape (t,3)
    # or
    # slm.lat  = 3D logical array
    # mask     = numpy 'bool' array of shape (1,v)
    
    slm_mat = slm.copy()
    for key in slm_mat.keys():
        if np.ndim(slm_mat[key]) == 0:
            slm_mat[key] = surfstat_eng.double(slm_mat[key].item())
        else:
            slm_mat[key] = matlab.double(slm_mat[key].tolist())

    # MATLAB errors if 'resl' is not provided and more than 1 output argument is requested.
    if 'resl' in 'slm':
        num_out = 3
    else:
        num_out = 1
    
    if mask is None:
        out = eng.SurfStatResels(slm_mat, 
                                nargout=num_out)
    else:
        mask_mat = matlab.double(np.array(mask, dtype=int).tolist())
        mask_mat = matlab.logical(mask_mat)
        out = eng.SurfStatResels(slm_mat, 
                                 mask_mat, 
                                 nargout=num_out)

    return np.array(out)


def dummy_test(slm, mask=None):

    # Run MATLAB
    try:
        mat_out = matlab_SurfStatResels(slm,mask)
        # Deal with either 1 or 3 output arguments.
        if isinstance(mat_out,tuple) and len(mat_out) == 3:
            mat_output = list(mat_out)
        else:
            mat_output = [mat_out]
    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # Run Python
    resels_py,  reselspvert_py,  edg_py =  py_SurfStatResels(slm,mask)
    if len(mat_output) == 1:
        py_output = [resels_py]
    else:
        py_output = [resels_py,
                     reselspvert_py,
                     edg_py]
    
    # compare matlab-python outputs
    test_out = []   
    for py, mat in zip(py_output,mat_output):
        result = np.allclose(np.squeeze(py),
                             np.squeeze(np.asarray(mat)),
                             rtol=1e-05, equal_nan=True)
        test_out.append(result)

    assert all(flag == True for (flag) in test_out)


# Test with only slm.tri
def test_1():
    slm = {'tri': np.array(
                  [[1,2,3],
                   [2,3,4], 
                   [1,2,4],
                   [2,3,5]])}
    dummy_test(slm)

# Test with slm.tri and slm.resl
def test_2():
    slm = {'tri': np.array(
               [[1,2,3],
                [2,3,4], 
                [1,2,4],
                [2,3,5]]),
          'resl': np.random.rand(8,6)}
    dummy_test(slm)

# Test with slm.tri, slm.resl, and mask
def test_3():
    slm = {'tri': np.array(
               [[1,2,3],
                [2,3,4], 
                [1,2,4],
                [2,3,5]]),
       'resl': np.random.rand(8,6)}
    mask = np.array([True,True,True,False,True])
    dummy_test(slm,mask)

# Test with slm.lat, 1's only.
def test_4():
    slm = {'lat': np.ones((10,10,10))}
    dummy_test(slm)

# Test with slm.lat, both 0's and 1's. 
def test_5():
    slm = {'lat': np.random.rand(10,10,10) > 0.5}
    dummy_test(slm)

# Test with slm.lat, both 0's and 1's, and a mask.
def test_6():
    slm = {'lat': np.random.rand(10,10,10) > 0.5}
    mask = np.random.choice([False,True],np.sum(slm['lat']))
    dummy_test(slm,mask)

# Test with slm.lat and slm.resl
def test_7():
    slm = {'lat': np.random.rand(10,10,10) > 0.5}
    edg = py_SurfStatEdg(slm)
    slm['resl'] = np.random.rand(edg.shape[0],1)
    dummy_test(slm)

# Test with slm.lat, slm.resl, and a mask
def test_8():
    slm = {'lat': np.random.rand(10,10,10) > 0.5}
    mask = np.random.choice([False,True],np.sum(slm['lat']))
    edg = py_SurfStatEdg(slm)
    slm['resl'] = np.random.rand(edg.shape[0],1)
    dummy_test(slm)
