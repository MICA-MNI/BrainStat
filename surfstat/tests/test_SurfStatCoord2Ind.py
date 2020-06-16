import sys
sys.path.append("python")
from SurfStatCoord2Ind import *
import surfstat_wrap as sw
import numpy as np

sw.matlab_init_surfstat()

def dummy_test(coord, surf):

    try:
        # wrap matlab functions
        Wrapped_ind = sw.matlab_SurfStatCoord2Ind(coord, surf)

    except:
        pytest.fail("ORIGINAL MATLAB CODE DOES NOT WORK WITH THESE INPUTS...")

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
    
    coord = np.array([[1,2,3], [4,5,6]])    
    surf = {}
    surf['lat'] = np.ones((10,10,10))
    dummy_test(coord, surf)
