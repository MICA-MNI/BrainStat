import sys
sys.path.append("python/need_not_convert")
from SurfStatCoord2Ind import *
import surfstat_wrap as sw
import numpy as np
import pytest

sw.matlab_init_surfstat()


def dummy_test(coord, surf):

    try:
        # wrap matlab functions
        Wrapped_ind = sw.matlab_SurfStatCoord2Ind(coord, surf)

    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    Python_ind = py_SurfStatCoord2Ind(coord, surf)

    # compare matlab-python outputs
    testout = np.allclose(Wrapped_ind, Python_ind, rtol=1e-05, equal_nan=True)
    assert testout


#### Test 1
def test_surf_coord():
    # coord is 2D array, surf['coord'] is 2D array
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)

    coord = np.random.rand(m,3)
    surf = {}
    surf['coord'] = np.random.rand(3,n)
    dummy_test(coord, surf)


### Test 2
def test_surf_lat_easy():
    # coord is 2D array, surf['lat'] is 3D array of ones
    m = np.random.randint(1,100)

    coord = np.random.rand(m,3)
    surf = {}
    surf['lat'] = np.ones((10,10,10))
    dummy_test(coord, surf)


### Test 3
def test_surf_lat_middle():
    # coord is 2D array, surf['lat'] is 3D array of ones and zeros
    m = np.random.randint(1,100)
    coord = np.random.rand(m,3)
    surf = {}
    surf['lat'] = np.random.choice([0, 1], size=(3,3,3))
    dummy_test(coord, surf)


### Test 4
def test_surf_lat_hard():
    # coord is 2D array, surf['lat'] is 3D array of ones and zeros
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)

    coord = np.array([[0.25684115, 0.45125523, 0.23784246],
       [0.99785513, 0.34429589, 0.97229819],
       [0.40949779, 0.56305147, 0.98313413],
       [0.41883601, 0.20865994, 0.08787483],
       [0.32232554, 0.21137128, 0.88918578]])
    surf = {}
    surf['lat'] = np.random.choice([0, 1], size=(n,k,l))
    dummy_test(coord, surf)


### Test 5
def test_surf_lat_complex():
    # coord is 2D array, surf['lat'] is 3D array of ones and zeros
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)

    coord = np.random.rand(m,3)
    surf = {}
    surf['lat'] = np.random.choice([0, 1], size=(n,k,l))
    dummy_test(coord, surf)


### Test 6
def test_surf_lat_vox_easy():
    # coord is 2D array, surf['lat'] is 3D array of ones, surf['vox'] is 2D
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)

    coord = np.random.rand(m,3)
    surf = {}
    surf['lat'] = np.ones((10,10,10))
    surf['vox'] = np.array([[n,k,l]])
    dummy_test(coord, surf)


### Test 7
def test_surf_lat_vox_middle():
    # coord is 2D array, surf['lat'] is 3D.. of ones & zeros, surf['vox'] is 2D
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)

    coord = np.random.rand(m,3)
    surf = {}
    surf['lat'] = np.random.choice([0, 1], size=(3,3,3))
    surf['vox'] = np.array([[n,k,l]])
    dummy_test(coord, surf)


### Test 8
def test_surf_lat_vox_complex():
    # coord is 2D array, surf['lat'] is 3D.. of ones & zeros, surf['vox'] is 2D
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)

    coord = np.random.rand(m,3)
    surf = {}
    surf['lat'] = np.random.choice([0, 1], size=(n,k,l))
    surf['vox'] = np.array([[l,k,n]])
    dummy_test(coord, surf)


### Test 9
def test_surf_lat_vox_origin_easy():
    # coord is 2D array, surf['lat'] 3D, surf['vox'] 2D, surf[orig] 2D
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)

    coord = np.random.rand(m,3)
    surf = {}
    surf['lat'] = np.ones((10,10,10))
    surf['vox'] = np.array([[n,k,l]])
    surf['origin'] = np.array([[k,l,n]])
    dummy_test(coord, surf)


### Test 10
def test_surf_lat_vox_origin_middle():
    # coord is 2D array, surf['lat'] 3D, surf['vox'] 2D, surf[orig] 2D
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)

    coord = np.random.rand(m,3)
    surf = {}
    surf['lat'] = np.random.choice([0, 1], size=(3,3,3))
    surf['vox'] = np.array([[n,k,l]])
    surf['origin'] = np.array([[k,l,n]])
    dummy_test(coord, surf)


### Test 11
def test_surf_lat_vox_origin_complex():
    # coord is 2D array, surf['lat'] 3D, surf['vox'] 2D, surf[orig] 2D
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)

    coord = np.random.rand(m,3)
    surf = {}
    surf['lat'] = np.random.choice([0, 1], size=(n,k,l))
    surf['vox'] = np.array([[l,k,n]])
    surf['origin'] = np.array([[k+5,l+5,n+5]])
    dummy_test(coord, surf)
