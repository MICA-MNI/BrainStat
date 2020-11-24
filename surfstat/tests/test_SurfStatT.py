import sys
sys.path.append("python")
from SurfStatT import *
from SurfStatLinMod import *
import surfstat_wrap as sw
import numpy as np
from scipy.io import loadmat
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
def test_01():
    a = np.random.randint(1,10)
    A = {}
    A['X']  = np.random.rand(a,1)
    A['df'] = np.array([[3.0]])
    A['coef'] = np.random.rand(1,a).reshape(1,a)
    A['SSE'] = np.random.rand(1, a)
    B = np.random.rand(1).reshape(1,1)

    dummy_test(A, B)


#### Test 2  ### square matrices
def test_02():
    a = np.random.randint(1,10)
    b = np.random.randint(1,10)
    A = {}
    A['X']    = np.random.rand(a,a)
    A['df']   = np.array([[b]])
    A['coef'] = np.random.rand(a,a)
    A['SSE']  = np.random.rand(1, a)
    B = np.random.rand(1, a)
    dummy_test(A, B)


#### Test 3  ### slm.V & slm.r given
def test_03():
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


#### Test 4 #### slm.V given, slm.r not
def test_04():
    A = {}
    A['X'] = np.random.rand(3,2)
    A['V'] = np.array([ [4,4,4], [5,5,5], [6,6,6] ])
    A['df'] = np.array([np.random.randint(1,10)])
    A['coef'] = np.random.rand(2,1)
    A['SSE'] = np.array([np.random.randint(1,10)])
    A['dr'] = np.array([np.random.randint(1,10)])
    B = np.array([[1]])
    dummy_test(A, B)


def test_05():
    fname = './tests/data/thickness_slm.mat'
    f = loadmat(fname)

    slm = {}
    slm['X'] = f['slm']['X'][0,0]
    slm['df'] = f['slm']['df'][0,0][0,0]
    slm['coef'] = f['slm']['coef'][0,0]
    slm['SSE'] = f['slm']['SSE'][0,0]
    slm['tri'] = f['slm']['tri'][0,0]
    slm['resl'] = f['slm']['resl'][0,0]

    AGE = f['slm']['AGE'][0,0]

    dummy_test(slm, AGE)


def test_06():
    fname = './tests/data/thickness_slm.mat'
    f = loadmat(fname)

    slm = {}
    slm['X'] = f['slm']['X'][0,0]
    slm['df'] = f['slm']['df'][0,0][0,0]
    slm['coef'] = f['slm']['coef'][0,0]
    slm['SSE'] = f['slm']['SSE'][0,0]
    slm['tri'] = f['slm']['tri'][0,0]
    slm['resl'] = f['slm']['resl'][0,0]

    AGE = f['slm']['AGE'][0,0]

    dummy_test(slm, -1*AGE)


def test_07():
    fname = './tests/data/thickness.mat'
    f = loadmat(fname)

    A = f['T']
    np.random.shuffle(A)

    AGE = Term(np.array(f['AGE']), 'AGE')
    B = 1 + AGE
    surf = {}
    surf['tri'] = f['tri']
    surf['coord'] = f['coord']
    slm = py_SurfStatLinMod(A, B, surf)

    contrast = np.array(f['AGE']).T

    dummy_test(slm, contrast)


def test_08():
    fname = './tests/data/sofopofo1_slm.mat'
    f = loadmat(fname)
    slm = {}
    slm['X'] = f['slm']['X'][0,0]
    slm['df'] = f['slm']['df'][0,0][0,0]
    slm['coef'] = f['slm']['coef'][0,0]
    slm['SSE'] = f['slm']['SSE'][0,0]
    slm['tri'] = f['slm']['tri'][0,0]
    slm['resl'] = f['slm']['resl'][0,0]

    contrast = np.random.randint(20,50, size=(slm['X'].shape[0],1))

    dummy_test(slm, contrast)


def test_09():
    fname = './tests/data/sofopofo1.mat'
    f = loadmat(fname)
    T = f['sofie']['T'][0,0]

    params = f['sofie']['model'][0,0]
    colnames = ['1', 'ak', 'female', 'male', 'Affect', 'Control1', 'Perspective',
    'Presence', 'ink']

    M = Term(params, colnames)

    SW = {}
    SW['tri'] = f['sofie']['SW'][0,0]['tri'][0,0]
    SW['coord'] = f['sofie']['SW'][0,0]['coord'][0,0]
    slm = py_SurfStatLinMod(T, M, SW)

    contrast = np.random.randint(20,50, size=(slm['X'].shape[0],1))

    dummy_test(slm, contrast)
