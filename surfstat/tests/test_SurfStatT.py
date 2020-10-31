import sys
sys.path.append("python")
from SurfStatT import *
import surfstat_wrap as sw
import numpy as np
import sys
import pytest

sw.matlab_init_surfstat()


def dummy_test(slm, contrast):

    try:
        # wrap matlab functions
        Wrapped_slm = sw.matlab_SurfStatT(slm, contrast)

    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # run python functions
    Python_slm = py_SurfStatT(slm, contrast)

    testout_SurfStatT = []

    # compare matlab-python outputs
    for key in Wrapped_slm:
        testout_SurfStatT.append(np.allclose(Python_slm[key], Wrapped_slm[key], \
                                 rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout_SurfStatT)


#### Test 1
def test_1d_row_vectors():
    a = np.random.randint(1,10)
    A = {}
    A['X']  = np.random.rand(a,1)
    A['df'] = np.array([[3.0]])
    A['coef'] = np.random.rand(1,a).reshape(1,a)
    A['SSE'] = np.random.rand(1, a)
    B = np.random.rand(1).reshape(1,1)

    dummy_test(A, B)


#### Test 2  ### square matrices
def test_2d_square_matrix():
    a = np.random.randint(1,10)
    b = np.random.randint(1,10)
    A = {}
    A['X']    = np.random.rand(a,a)
    A['df']   = np.array([[b]])
    A['coef'] = np.random.rand(a,a)
    A['SSE']  = np.random.rand(1, a)
    B = np.random.rand(1, a)
    dummy_test(A, B)


#### Test 3a  ### slm.V & slm.r given
def test_2d_fullslm():
    a = np.array([ [4,4,4], [5,5,5], [6,6,6] ])
    b = np.array([[1,0,0], [0,1,0], [0,0,1]])
    Z = np.zeros((3,3,2))
    Z[:,:,0] = a
    Z[:,:,1] = b
    A = {}
    A['X'] = np.array([[1, 2], [3,4], [5,6]])
    A['V'] = Z
    A['df'] = np.array([[1.0]])
    A['coef'] = np.array([[8] , [9]])
    A['SSE'] = np.array([[3]])
    A['r'] = np.array([[4]])
    A['dr'] = np.array([[5]])
    B = np.array([[1]])
    dummy_test(A, B)


#### Test 3b #### slm.V given, slm.r not
def test_2d_partial_slm():
    A = {}
    A['X'] = np.random.rand(3,2)
    A['V'] = np.array([ [4,4,4], [5,5,5], [6,6,6] ])
    A['df'] = np.array([np.random.randint(1,10)])
    A['coef'] = np.random.rand(2,1)
    A['SSE'] = np.array([np.random.randint(1,10)])
    A['dr'] = np.array([np.random.randint(1,10)])
    B = np.array([[1]])
    dummy_test(A, B)
