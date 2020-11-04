import sys
sys.path.append("python")
from SurfStatNorm import *
import surfstat_wrap as sw
import numpy as np
import pytest

sw.matlab_init_surfstat()


def dummy_test(Y, mask, subdiv):

    try:
        # wrap matlab functions
        Wrapped_Y, Wrapped_Yav = sw.matlab_SurfStatNorm(Y, mask, subdiv)
    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    Python_Y, Python_Yav = py_SurfStatNorm(Y, mask, subdiv)

    # compare matlab-python outputs

    assert np.allclose(Wrapped_Y, Python_Y, rtol=1e-05, equal_nan=True)

    assert np.allclose(Wrapped_Yav, Python_Yav, rtol=1e-05, equal_nan=True)


#### test 1a
# 1D inputs --- row vectors
def test_1d_row_vectors():
    v = np.random.randint(1,9)

    a = np.arange(1,v)
    a = a.reshape(1, len(a))

    Y = a
    mask = None
    subdiv = 's'

    dummy_test(Y, mask, subdiv)


#### test 1b
# 1D inputs --- row vectors & mask
def test_1d_row_vectors_mask():
    a = np.arange(1,11)
    a = a.reshape(1, len(a))
    Y = a
    mask = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    subdiv = 's'
    dummy_test(Y, mask, subdiv)


#### test 2a
# 1D inputs --- 2D arrays & mask
def test_1d_2d_array_vectors_mask():
    a = np.arange(1,11)
    a = a.reshape(1, len(a))
    Y = np.concatenate((a,a), axis=0)
    mask = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
    subdiv = 's'
    dummy_test(Y, mask, subdiv)


#### test 3a
# 1D inputs --- 3D arrays & mask
def test_1d_3d_array_mask():

    a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    Y = np.zeros((3,4,2))
    Y[:,:,0] = a
    Y[:,:,1] = a
    mask = np.array([1, 1, 0, 0], dtype=bool)
    subdiv = 's'
    dummy_test(Y, mask, subdiv)
