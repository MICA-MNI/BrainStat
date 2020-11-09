import sys
sys.path.append("python")
from SurfStatSmooth import *
import surfstat_wrap as sw
import numpy as np
import pytest

sw.matlab_init_surfstat()


def dummy_test(Y, surf, FWHM):

    try:
        # wrap matlab functions
        Wrapped_Y = sw.matlab_SurfStatSmooth(Y, surf, FWHM)
    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # run matlab equivalent
    Python_Y = py_SurfStatSmooth(Y, surf, FWHM)

    # compare matlab-python outputs
    assert np.allclose(Wrapped_Y, Python_Y, rtol=1e-05, equal_nan=True)


def test_01():
    n = np.random.randint(1,100)
    Y = np.random.rand(n,n)
    surf = {}
    surf['tri'] = np.array([[1,2,3]])
    FWHM = 3.0
    dummy_test(Y, surf, FWHM)


def test_02():
    n = np.random.randint(1,100)
    Y = np.random.rand(n,n)
    m = np.random.randint(1,100)
    surf = {}
    surf['tri'] = np.random.randint(1,20,size=(m,3))
    FWHM = 3.0
    dummy_test(Y, surf, FWHM)


def test_03():
    n = np.random.randint(1,100)
    Y = np.random.rand(n,n)
    m = np.random.randint(1,100)
    surf = {}
    surf['lat'] = np.random.choice([0, 1], size=(3,3,3))
    FWHM = 3.0
    dummy_test(Y, surf, FWHM)


def test_04():
    n = np.random.randint(1,100)
    a = np.random.rand(n,3)
    b = np.random.rand(n,3)
    Y = np.zeros((n,3,2))
    Y[:,:,0] = a
    Y[:,:,1] = b
    surf = {}
    surf['tri'] = np.array([[1,2,3]])
    FWHM = 3.0
    dummy_test(Y, surf, FWHM)
