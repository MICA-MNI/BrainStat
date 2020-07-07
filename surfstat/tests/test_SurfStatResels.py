import sys
sys.path.append("python")
from SurfStatResels import py_SurfStatResels
import numpy as np
import matlab.engine
import math
import itertools
import pytest

eng = matlab.engine.start_matlab()
eng.addpath('matlab/')


def py_array_to_mat(L):
    L = np.array(L) # Ascertain input is a numpy array. 
    S = L.shape
    if L.ndim == 1:
        S = [S[0],1]
        L = L.tolist()
    else:
        S = list(S)
        L = [item for sublist in L.tolist() for item in sublist]
    S.reverse()
    S = eng.cell2mat(S)
    M = eng.cell2mat(L)
    return eng.transpose(eng.reshape(M,S))


def matlab_SurfStatResels(slm, mask=None): 
    # slm.resl = numpy array of shape (e,k)
    # slm.tri  = numpy array of shape (t,3)
    # or
    # slm.lat  = 3D logical array
    # mask     = numpy 'bool' array of shape (1,v)
    
    slm_mat = slm.copy()
    for key in slm_mat.keys():
        slm_mat[key] = py_array_to_mat(slm_mat[key])

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
    slm = {'tri': [[1,2,3],
                   [2,3,4], 
                   [1,2,4],
                   [2,3,5]]}
    dummy_test(slm)

# Test with slm.tri and slm.resl
def test_2():
    slm = {'tri':  [[1,2,3],
                [2,3,4], 
                [1,2,4],
                [2,3,5]],
          'resl': np.random.rand(8,6)}
    dummy_test(slm)

# Test with slm.tri, slm.resl, and mask
def test_3():
    slm = {'tri':  [[1,2,3],
                [2,3,4], 
                [1,2,4],
                [2,3,5]],
       'resl': np.random.rand(8,6)}
    mask = np.array([True,True,True,False,True])
    dummy_test(slm,mask)

# Add a tri/resl/mask test with real data. 