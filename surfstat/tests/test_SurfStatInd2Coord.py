import sys
sys.path.append("python")
from SurfStatInd2Coord import *
import surfstat_wrap as sw
import numpy as np

sw.matlab_init_surfstat()

def dummy_test(A, B):

    try:
        # wrap matlab functions
        Wrapped_coord = sw.matlab_SurfStatInd2Coord(A, B)

    except:
        pytest.fail("ORIGINAL MATLAB CODE DOES NOT WORK WITH THESE INPUTS...")

    Python_coord = py_SurfStatInd2Coord(A, B)

    # compare matlab-python outputs
    testout = np.allclose(Wrapped_coord, Python_coord, rtol=1e-05, equal_nan=True)

    assert testout

#### Test 1 
def test_surf_coord():
    # A is 2D array, B['coord'] is 2D array
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)

    A = np.ones((1,m))
    B = {}
    B['coord'] = np.random.rand(3,n)
    dummy_test(A, B)

### Test 2 
def test_surf_lat_easy():
    # coord is 2D array, surf['lat'] is 3D array of ones
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
 
    A = np.ones((1,m))
    B = {}
    B['lat'] = np.ones((n,n,n))
    dummy_test(A, B)

### Test 3 
def test_surf_lat_hard():
    # coord is 2D array, surf['lat'] is 3D array of ones
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
 
    A = np.ones((1,m))
    B = {}
    B['lat'] = np.random.choice([0, 1], size=(n,n,n))
    dummy_test(A, B)

### Test 4 
def test_surf_lat_complex():
    # coord is 2D array, surf['lat'] is 3D array of ones
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)
 
    A = np.ones((1,m))
    B = {}
    B['lat'] = np.random.choice([0, 1], size=(n,l,k))
    dummy_test(A, B)

### Test 5 
def test_surf_lat_vox_easy():
    # coord is 2D array, surf['lat'] is 3D array of ones, surf['vox'] is 2D array
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)

    A = np.ones((1,m))
    B = {}
    B['lat'] = np.random.choice([0, 1], size=(n,n,n))
    B['vox'] = np.array([[n,k,l]])
    dummy_test(A, B)

### Test 6 
def test_surf_lat_vox_complex():
    # coord is 2D array, surf['lat'] is 3D array of ones
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)
 
    A = np.ones((1,m))
    B = {}
    B['lat'] = np.random.choice([0, 1], size=(n,l,k))
    B['vox'] = np.array([[n,k,l]])
    dummy_test(A, B)

### Test 7
def test_surf_lat_vox_origin_easy():
    # coord is 2D array, surf['lat'] 3D, surf['vox'] 2D, surf[orig] 2D 
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)
 
    A = np.ones((1,m))
    B = {}
    B['lat'] = np.random.choice([0, 1], size=(n,n,n))
    B['vox'] = np.array([[n,k,l]])
    B['origin'] = np.array([[k,l,n]])
    dummy_test(A, B)

### Test 8
def test_surf_lat_vox_origin_hard():
    # coord is 2D array, surf['lat'] 3D, surf['vox'] 2D, surf[orig] 2D 
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)
 
    A = np.ones((1,m))
    B = {}
    B['lat'] = np.random.choice([0, 1], size=(k,n,l))
    B['vox'] = np.array([[n,k,l]])
    B['origin'] = np.array([[k,l,n]])
    dummy_test(A, B)

