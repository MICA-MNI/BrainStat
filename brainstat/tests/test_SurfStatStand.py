from brainstat.stats import *
import surfstat_wrap as sw
import numpy as np
import pytest

sw.matlab_init_surfstat()


def dummy_test(Y, mask, subtractordivide):

    try:
        # wrap matlab functions
        Wrapped_Y, Wrapped_Ym = sw.matlab_Stand(Y, mask, subtractordivide)

    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")


    # python function
    Python_Y, Python_Ym = SurfStatStand(Y, mask, subtractordivide)

    # compare matlab-python outputs
    testout_Stand = []

    testout_Stand.append(np.allclose(Wrapped_Y, Python_Y, \
                                 rtol=1e-05, equal_nan=True))

    testout_Stand.append(np.allclose(Wrapped_Ym, Python_Ym, \
                                 rtol=1e-05, equal_nan=True))
    #result_Stand = all(flag == True for (flag) in testout_Stand)

    assert all(flag == True for (flag) in testout_Stand)


def test_01():
    # 1D inputs --- row vectors
    v = np.random.randint(1,9)
    a = np.arange(1,v)
    a = a.reshape(1, len(a))
    Y = a
    mask = None
    subtractordivide = 's'
    dummy_test(Y, mask=mask, subtractordivide=subtractordivide)


def test_02():
    # 1D inputs --- row vectors & mask
    a = np.arange(1,11)
    a = a.reshape(1, len(a))
    Y = a
    mask = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    subtractordivide = 's'
    dummy_test(Y, mask=mask, subtractordivide=subtractordivide)


def test_03():
    # 2D inputs --- 2D arrays & mask
    a = np.arange(1,11)
    a = a.reshape(1, len(a))
    Y = np.concatenate((a,a), axis=0)
    mask = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
    subtractordivide = 's'
    dummy_test(Y, mask=mask, subtractordivide=subtractordivide)


def test_04():
    # 3D inputs --- 3D arrays & mask
    a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    Y = np.zeros((3,4,2))
    Y[:,:,0] = a
    Y[:,:,1] = a
    mask = np.array([1, 1, 0, 0], dtype=bool)
    subtractordivide = 's'
    dummy_test(Y, mask=mask, subtractordivide=subtractordivide)
