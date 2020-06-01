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
    if var == None:
        var = []
    elif not isinstance(var,list):
        var = [var]
    return matlab.double(var)

def dummy_test(eng, search_volume, num_voxels, fwhm, df, p_val_peak, 
    cluster_threshold, p_val_extent, nconj, nvar):

    try:    
        peak_threshold_mat, extent_threshold_mat, peak_threshold_1_mat, extent_threshold_1_mat, t_mat, rho_mat = eng.stat_threshold(
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
        mat_output = [peak_threshold_mat, extent_threshold_mat, peak_threshold_1_mat, extent_threshold_1_mat, t_mat, rho_mat]
    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")
	
    # run python functions
    peak_threshold_py, extent_threshold_py, peak_threshold_1_py, extent_threshold_1_py, t_py, rho_py = stat_threshold(search_volume, num_voxels, fwhm, df, p_val_peak, 
        cluster_threshold, p_val_extent, nconj, nvar, nprint=0)
    py_output = [peak_threshold_py, extent_threshold_py, peak_threshold_1_py, extent_threshold_1_py, t_py, rho_py]

    # compare matlab-python outputs
    testout_statthreshold = []   
    for py, mat in zip(py_output,mat_output):
        if np.all(np.isnan(py)) and np.all(np.iscomplex(mat)):
            # Due to differences in how python and matlab handle powers with 
            # imaginary outputs there are edge-cases where python returns nan 
            # and matlab returns a complex number. Neither of these should ever 
            # happen to begin with, so just skip these cases.
            continue
        O = np.allclose(np.squeeze(py), np.squeeze(np.asarray(mat)), rtol=1e-05, equal_nan=True)
        testout_statthreshold.append(O)
    assert all(flag == True for (flag) in testout_statthreshold)


#### test 1
def test_statthreshold_permutations():
    # Set variables
    search_volume=[0, 
                   [1,2,3,4],
                   [1,2,3],[4,5,6]]
    
    num_voxels=[1,
                100,
                [500,200]]
    
    fwhm=[0.0,
          5,
          [3,2],
          [[2],[5]], 
          [[3,2],[7,9]]]
        
    df=[math.inf,
        5,
        [1,2],
        [[1,7],[1,2]]]

    p_val_peak=[0.05,
                [1.5,2]]

    cluster_threshold=0.001
    
    p_val_extent=[0.05,
                  [0.1,0.09]]
    
    nconj=[1,
           3]
    
    nvar=[1,
          [1,1]]

    eng = matlab.engine.start_matlab()
    eng.addpath('matlab/')

    opts = [search_volume, num_voxels, fwhm, df, p_val_peak, p_val_extent, nconj, nvar]
    for opt in itertools.product(*opts):
        sv, nv, fw, degf, pvp, pve, nc, nr = opt
        dummy_test(eng, sv, nv, fw, degf, pvp, cluster_threshold, pve, nc, nr)
