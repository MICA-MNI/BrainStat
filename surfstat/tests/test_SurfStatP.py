import sys
sys.path.append("python")
from SurfStatP import *
import surfstat_wrap as sw
from scipy.io import loadmat
import numpy as np
import pytest

def dummy_test(slm, mask=None, clusthresh=0.001):

    try:
        # wrap matlab functions
        M_pval, M_peak, M_clus, M_clusid = sw.matlab_SurfStatP(slm, 
                                                               mask, 
                                                               clusthresh)

    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # run python equivalent
    PY_pval, PY_peak, PY_clus, PY_clusid = py_SurfStatP(slm, 
                                                        mask,
                                                        clusthresh)

    # compare matlab-python outputs
    testout_SurfStatP = []

    for key in M_pval:
        testout_SurfStatP.append(np.allclose(M_pval[key], PY_pval[key], 
                                      rtol=1e-05, equal_nan=True))
    for key in M_peak:
        testout_SurfStatP.append(np.allclose(M_peak[key], PY_peak[key], 
                                      rtol=1e-05, equal_nan=True))
    for key in M_clus:
        testout_SurfStatP.append(np.allclose(M_clus[key], PY_clus[key], 
                                      rtol=1e-05, equal_nan=True))
    testout_SurfStatP.append(np.allclose(M_clusid, PY_clusid, 
                              rtol=1e-05, equal_nan=True))


    assert (all(flag == True for (flag) in testout_SurfStatP))

def test_1():
    # special case, v =1, l=1
    l = int(1)
    v = int(1)
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = np.random.rand(l,v)
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    slm['dfs'] =  np.array([[np.random.randint(1,10)]])
    dummy_test(slm)
