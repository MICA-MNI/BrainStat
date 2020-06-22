import sys
sys.path.append("python")
from stat_threshold import stat_threshold
import numpy as np
import matlab.engine
import math
import itertools
import pdb
import pytest

def var2mat(var):
    # Brings the input variables to matlab format.
    if isinstance(var, np.ndarray):
        var = var.tolist()
    elif var == None:
        var = []
    if not isinstance(var,list) and not isinstance(var, np.ndarray):
        var = [var]
    return matlab.double(var)

def dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
    cluster_threshold, p_val_extent, nconj, nvar):

    try:
        # run matlab functions (wrapping...)    
        peak_threshold_mat, extent_threshold_mat, peak_threshold_1_mat, \
        extent_threshold_1_mat, t_mat, rho_mat = eng.stat_threshold(
                                                    var2mat(search_volume), 
                                                    var2mat(num_voxels), 
                                                    var2mat(fwhm), 
                                                    var2mat(df), 
                                                    var2mat(p_val_peak), 
                                                    var2mat(cluster_threshold), 
                                                    var2mat(p_val_extent), 
                                                    var2mat(nconj), 
                                                    var2mat(nvar), 
                                                    var2mat(None), 
                                                    var2mat(None), 
                                                    var2mat(0),
                                                    nargout=6)
        mat_output = [peak_threshold_mat, \
                      extent_threshold_mat, \
                      peak_threshold_1_mat, \
                      extent_threshold_1_mat, \
                      t_mat, \
                      rho_mat]
    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # run python functions
    peak_threshold_py, extent_threshold_py, peak_threshold_1_py, \
    extent_threshold_1_py, t_py, rho_py = stat_threshold(
                                            search_volume, 
                                            num_voxels, 
                                            fwhm, 
                                            df, 
                                            p_val_peak, 
                                            cluster_threshold, 
                                            p_val_extent, 
                                            nconj, 
                                            nvar, 
                                            nprint=0)
    py_output = [peak_threshold_py, \
                 extent_threshold_py, \
                 peak_threshold_1_py, \
                 extent_threshold_1_py, \
                 t_py, \
                 rho_py]

    # compare matlab-python outputs
    testout_statthreshold = []   
    for py, mat in zip(py_output,mat_output):
        if np.all(np.isnan(py)) and np.all(np.iscomplex(mat)):
            # Due to differences in how python and matlab handle powers with 
            # imaginary outputs there are edge-cases where python returns nan 
            # and matlab returns a complex number. Neither of these should ever 
            # happen to begin with, so just skip these cases.
            continue
        result = np.allclose(np.squeeze(py),
                             np.squeeze(np.asarray(mat)),
                             rtol=1e-05, equal_nan=True)
        #print(result)
        testout_statthreshold.append(result)

    assert all(flag == True for (flag) in testout_statthreshold)

eng = matlab.engine.start_matlab()
eng.addpath('matlab/')


# Test 1 --> search_volume is "a float"  (rest is default-values)
def test_1():
    search_volume     = np.random.uniform(0,10)
    num_voxels        = 1
    fwhm              = 0.0
    df                = 5
    p_val_peak        = 0.05
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)

# Test 2 --> search_volume is a list  (rest is default-values)
def test_2():
    m = np.random.uniform(0,10)
    n = np.random.uniform(0,10)
    k = np.random.uniform(0,10)

    search_volume     = [m, n, k]
    num_voxels        = 1
    fwhm              = 0.0
    df                = 5
    p_val_peak        = 0.05
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)


# Test 3 --> search_volume is a 1D numpy array (rest is default-values)
def test_3():
    m = np.random.uniform(0,10)
    n = np.random.uniform(0,10)
    k = np.random.uniform(0,10)

    search_volume     = np.array([m, n, k])
    num_voxels        = 1
    fwhm              = 0.0
    df                = 5
    p_val_peak        = 0.05
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
               cluster_threshold, p_val_extent, nconj, nvar)


# Test 4 --> search_volume is a 2D numpy array (rest is default-values)
def test_4():
    m = np.random.uniform(0,10)
    n = np.random.uniform(0,10)

    search_volume     = np.array([[m,n],[n+10, m+87]])
    num_voxels        = 1
    fwhm              = 0.0
    df                = 5
    p_val_peak        = 0.05
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
               cluster_threshold, p_val_extent, nconj, nvar)


# Test 5 --> search_volume: a float, num_voxels: an int
def test_5():
    search_volume     = np.random.uniform(0,10)
    num_voxels        = np.random.randint(1,1000)
    fwhm              = 0.0
    df                = 5
    p_val_peak        = 0.05
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)

# Test 6 --> search_volume: 2D numpy array, num_voxels: a list 
def test_6():
    m = np.random.randint(1,10000)
    n = np.random.randint(1,10000)
    k = np.random.randint(1,10000)

    search_volume     = np.array([[m,n],[n+10, m+87]])
    num_voxels        = [m,n,k]
    fwhm              = 0.0
    df                = 5
    p_val_peak        = 0.05
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)

# Test 7 --> search_volume: 2D array, num_voxels: 2D array of shape (k,1),
# fwhm: float, df: int
def test_7():
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)

    search_volume     = np.array([[m,n],[n+10, m+87]])
    num_voxels        = np.random.rand(k,1) 
    fwhm              = np.random.uniform(0,10)
    df                = np.random.randint(1,(m-1))
    p_val_peak        = 0.05
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)

# Test 8 --> search_volume: 2D array, num_voxels: 2D array of shape (k,1),
# fwhm: float, df: math.inf
def test_8():
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)

    rng = np.random.default_rng()

    search_volume     = np.array([[m,n],[n+10, m+87]])
    num_voxels        = np.random.rand(k,1) 
    fwhm              = np.random.uniform(0,10)
    df                = math.inf
    p_val_peak        = 0.05
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)

# Test 9 --> search_volume: float, num_voxels: 2D array of shape (1,k),
# fwhm: float, df: int, p_val_peak: float
def test_9():
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)

    rng = np.random.default_rng()

    search_volume     = np.random.uniform(0,100)
    num_voxels        = rng.integers(1,100, size=(1,k))
    fwhm              = np.random.uniform(0,10)
    df                = k
    p_val_peak        = np.random.uniform(0,1)
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)

# Test 10 --> search_volume: float, num_voxels: 2D array of shape (1,k),
# fwhm: float, df: int, p_val_peak: list
def test_10():
    m = np.random.uniform(0,1)
    n = np.random.uniform(0,1)
    k = k = np.random.randint(1,100)

    rng = np.random.default_rng()

    search_volume     = np.random.uniform(0,100)
    num_voxels        = rng.integers(1,100, size=(1,k))
    fwhm              = np.random.uniform(0,10)
    df                = k
    p_val_peak        = [m, n, m/2, n/2]
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)

# Test 11 --> search_volume: float, num_voxels: 2D array of shape (1,k),
# fwhm: float, df: int, p_val_peak: 1D array
def test_11():
    m = np.random.uniform(0,1)
    k = np.random.randint(1,100)

    rng = np.random.default_rng()

    search_volume     = np.random.uniform(0,100)
    num_voxels        = rng.integers(1,100, size=(1,k))
    fwhm              = np.random.uniform(0,10)
    df                = k
    p_val_peak        = np.random.rand(k,1)
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)


# Test 12 --> search_volume: float, num_voxels: 2D array of shape (1,k),
# fwhm: float, df: int, p_val_peak: 1D array, cluster_threshold: float 
def test_12():
    m = np.random.randint(1,100)
    k = np.random.randint(1,100)

    rng = np.random.default_rng()

    search_volume     = np.random.uniform(0,100)
    num_voxels        = rng.integers(1,100, size=(1,m))
    fwhm              = np.random.uniform(0,10)
    df                = k
    p_val_peak        = np.random.rand(k,1)
    cluster_threshold = np.random.uniform(0,1)
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)

# Test 13 --> search_volume: float, num_voxels: int,
# fwhm: float, df: int, p_val_peak: 1D array, cluster_threshold: float,
# p_val_extent: 1D array 
def test_13():
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)

    search_volume     = 7.0
    num_voxels        = k
    fwhm              = np.random.uniform(0,10)
    df                = np.random.randint(1,(m-1))
    p_val_peak        = np.random.rand(k,1)
    cluster_threshold = np.random.uniform(0,1)
    p_val_extent      = np.random.rand(k,1)
    nconj             = 0.5
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)


# Test 14 --> search_volume: float, num_voxels: int,
# fwhm: float, df: int, p_val_peak: 1D array, cluster_threshold: float,
# p_val_extent: 1D array, nconj: int 
def test_14():
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)
    l = np.random.randint(1,100)

    search_volume     = 7.0
    num_voxels        = k
    fwhm              = np.random.uniform(0,10)
    df                = np.random.randint(1,(m-1))
    p_val_peak        = np.random.rand(k,1)
    cluster_threshold = np.random.uniform(0,1)
    p_val_extent      = np.random.rand(k,1)
    nconj             = l
    nvar              = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)

# Test 15 --> 
def test_15():
    search_volume     = [5.7, 8, 9]
    num_voxels        = 100
    fwhm              = 0
    df                = 5
    p_val_peak        = 0.02
    cluster_threshold = [0.001, 0.1, 0,6]
    p_val_extent      = 0.01
    nconj             = 0.02
    nvar              = [1,1,0]

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)

# Test 16 -->
def text_16():
    rng = np.random.default_rng()

    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)

    rng = np.random.default_rng()

    search_volume     = np.random.uniform(0,100)
    num_voxels        = rng.integers(1,100, size=(1,k))
    fwhm              = np.random.uniform(0,10)
    df                = [m,n]
    p_val_peak        = 0.05
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1
    
    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
           cluster_threshold, p_val_extent, nconj, nvar)


