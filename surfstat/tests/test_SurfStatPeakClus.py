import sys
sys.path.append("python")
from SurfStatPeakClus import *
import surfstat_wrap as sw
import numpy as np
import pytest


def dummy_test(slm, mask, thresh, reselspvert=None, edg=None):
    try:
        # wrap matlab functions
        M_peak, M_clus, M_clusid = sw.matlab_SurfStatPeakClus(slm, mask, 
                                                              thresh, 
                                                              reselspvert, edg)
    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # call python functions
    P_peak, P_clus, P_clusid = py_SurfStatPeakClus(slm, mask, thresh, 
                                                   reselspvert, edg)
    # compare matlab-python outputs
    testout_SurfStatPeakClus = []
    for key in M_peak:
        testout_SurfStatPeakClus.append(np.allclose(M_peak[key], P_peak[key], 
                                        rtol=1e-05, equal_nan=True))
    for key in M_clus:
        testout_SurfStatPeakClus.append(np.allclose(M_clus[key], P_clus[key], 
                                        rtol=1e-05, equal_nan=True))
    testout_SurfStatPeakClus.append(np.allclose(M_clusid, P_clusid, 
                                    rtol=1e-05, equal_nan=True))
    
    #print('XXXXXX ', all(flag == True for (flag) in testout_SurfStatPeakClus ))

    assert all(flag == True for (flag) in testout_SurfStatPeakClus)
    #return


sw.matlab_init_surfstat()

def test_1():
    # data from Sofie
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = slmdata['slm']['t'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    mask = np.ones((1,64984))
    thresh = 0.2
    
    dummy_test(slm, mask, thresh)


