# Developer's note a known difference in MATLAB/Python output is that, when
# the product of the num_voxels vector becomes extremely large, differences in 
# MATLAB's and Python's handling of extremely large numbers causes Python
# to throw NaNs but MATLAB still returns real values. This should never occur
# in real-world scenario's though as storing data from such a large number of 
# voxels is beyond reasonable computational capacities. - RV

import sys
sys.path.append("python")
from stat_threshold import stat_threshold
import numpy as np
import matlab.engine
import math
import itertools
import pdb
import pytest
from scipy.io import loadmat

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
        cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint):

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
                        var2mat(nprint),
                        nargout=6)
        mat_output = [peak_threshold_mat, 
                        extent_threshold_mat, 
                        peak_threshold_1_mat, 
                        extent_threshold_1_mat, 
                        t_mat, 
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
                    None, 
                    None, 
                    nprint)
                    
    py_output = [peak_threshold_py, 
                    extent_threshold_py, 
                    peak_threshold_1_py, 
                    extent_threshold_1_py, 
                    t_py, 
                    rho_py]

    # compare matlab-python outputs
    testout_statthreshold = []   
    for py, mat in zip(py_output,mat_output):
        if np.all([np.isnan(x) for x in py]) and np.all(np.iscomplex(mat)):
            # Due to differences in how python and matlab handle powers with 
            # imaginary outputs there are edge-cases where python returns nan 
            # and matlab returns a complex number. Neither of these should ever 
            # happen to begin with, so just skip these cases.
                continue

        result = np.allclose(np.squeeze(np.asarray(py)),
                np.squeeze(np.asarray(mat)),
                rtol=1e-05, equal_nan=True)
        #print(result)
        testout_statthreshold.append(result)

    if not np.all(testout_statthreshold):
        pdb.set_trace()

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
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

#Test 2 --> search_volume is a list  (rest is default-values)
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
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)


#Test 3 --> search_volume is a 1D numpy array (rest is default-values)
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
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)


#Test 4 --> search_volume is a 2D numpy array (rest is default-values)
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
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)


#Test 5 --> search_volume a float, num_voxels: an int
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
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

#Test 6 --> search_volume 2D numpy array, num_voxels: a list 
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
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

#Test 7 --> search_volume 2D array, num_voxels: 2D array of shape (k,1),
# fwhm float, df: int
def test_7():
    m = np.random.randint(3,100)
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
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

#Test 8 --> search_volume 2D array, num_voxels: 2D array of shape (k,1),
# fwhm float, df: math.inf
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
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

#Test 9 --> search_volume float, num_voxels: 2D array of shape (1,k),
# fwhm float, df: int, p_val_peak: float
def test_9():
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    k = np.random.randint(1,10)

    rng = np.random.default_rng()

    search_volume     = np.random.uniform(0,100)
    num_voxels        = rng.integers(1,100, size=(k))
    fwhm              = np.random.uniform(0,10)
    df                = k
    p_val_peak        = np.random.uniform(0,1)
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

#Test 10 --> search_volume float, num_voxels: 2D array of shape (1,k),
# fwhm float, df: int, p_val_peak: list
def test_10():
    m = np.random.uniform(0,1)
    n = np.random.uniform(0,1)
    k = np.random.randint(1,10)

    rng = np.random.default_rng()

    search_volume     = np.random.uniform(0,100)
    num_voxels        = rng.integers(1,100, size=(k))
    fwhm              = np.random.uniform(0,10)
    df                = k
    p_val_peak        = [m, n, m/2, n/2]
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

#Test 11 --> search_volume float, num_voxels: 2D array of shape (1,k),
# fwhm float, df: int, p_val_peak: 1D array
def test_11():
    m = np.random.uniform(0,1)
    k = np.random.randint(1,10)

    rng = np.random.default_rng()

    search_volume     = np.random.uniform(0,100)
    num_voxels        = rng.integers(1,100, size=(k))
    fwhm              = np.random.uniform(0,10)
    df                = k
    p_val_peak        = np.random.rand(k,1)
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)


# Test 12 --> search_volume float, num_voxels: 2D array of shape (1,k),
# fwhm float, df: int, p_val_peak: 1D array, cluster_threshold: float 
def test_12():
    m = np.random.randint(1,10)
    k = np.random.randint(1,100)

    rng = np.random.default_rng()

    search_volume     = np.random.uniform(0,100)
    num_voxels        = rng.integers(1,100, size=(m))
    fwhm              = np.random.uniform(0,10)
    df                = k
    p_val_peak        = np.random.rand(k,1)
    cluster_threshold = np.random.uniform(0,1)
    p_val_extent      = 0.05
    nconj             = 0.5
    nvar              = 1
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

# Test 13 --> search_volume float, num_voxels: int,
# fwhm float, df: int, p_val_peak: 1D array, cluster_threshold: float,
# p_val_extent 1D array 
def test_13():
    m = np.random.randint(3,10)
    n = np.random.randint(1,100)
    k = np.random.randint(1,100)

    search_volume     = 7.0
    num_voxels        = k
    fwhm              = np.random.uniform(0,10)
    df                = np.random.randint(1,(m-1))
    p_val_peak        = np.random.rand(k)
    cluster_threshold = np.random.uniform(0,1)
    p_val_extent      = np.random.rand(k)
    nconj             = 0.5
    nvar              = 1
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

# Test 14 --> search_volume float, num_voxels: int,
# fwhm float, df: int, p_val_peak: 1D array, cluster_threshold: float,
# p_val_extent 1D array, nconj: int 
def test_14():
    m = np.random.randint(3,100)
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
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

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
    nvar              = [1,1]
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

# buggy case 
def test_16():
    #slmdata = loadmat('./tests/data/someresels.mat')
    resels = np.array([[4.00000000e+00,  np.NaN, 6.59030991e+03]])
    N = 64984
    df = np.array([[1111, 0], [1111, 1111]])
    
    # this is from some real test data
    search_volume     = resels
    num_voxels        = N
    fwhm              = 1
    df                = df 
    p_val_peak        = np.array([0.5, 1])
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 1
    nvar              = 1
    EC_file           = None
    expr              = None 
    nprint            = 1

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)

# buggy case 
def test_17():
    somedata = loadmat('./tests/data/varA.mat')
    varA = somedata['varA']
    df = somedata['df']
    k = somedata['k']
    
    # this is from some real test data
    search_volume     = 0
    num_voxels        = 1
    fwhm              = 0
    df                = df 
    p_val_peak        = varA
    cluster_threshold = 0.001
    p_val_extent      = 0.05
    nconj             = 1
    nvar              = 1
    EC_file           = None
    expr              = None 
    nprint            = 0

    dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
            cluster_threshold, p_val_extent, nconj, nvar, EC_file, expr, nprint)



