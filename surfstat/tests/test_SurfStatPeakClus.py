import sys
sys.path.append("python")
from SurfStatPeakClus import *
from SurfStatEdg import *
import surfstat_wrap as sw
import numpy as np
import random
from scipy.io import loadmat
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
    
    assert all(flag == True for (flag) in testout_SurfStatPeakClus)

sw.matlab_init_surfstat()

def test_1():
    # data from Sofie, randomize threshold between 0 and 1
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = slmdata['slm']['t'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    mask = np.ones((1,64984))
    thresh = random.random()
    dummy_test(slm, mask, thresh)

def test_2():
    # generate random data, small sized as 1000 points for slm['t']
    k = 1000
    m = 100
    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    mask = np.ones((1,k))
    thresh = random.random()
    dummy_test(slm, mask, thresh)
    
def test_3():
    # generate random data, size will be also random
    k = random.randint(100, 1000)
    m = random.randint(100, 1000)
    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    mask = np.ones((1,k))
    thresh = random.random()
    dummy_test(slm, mask, thresh)
    
def test_4():
    # generate random data, size will be also random, big sizes...
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    mask = np.ones((1,k))
    thresh = random.random()
    dummy_test(slm, mask, thresh)
    
def test_5():
    # data from Sofie, randomize threshold between 0 and 1, add reselspvert
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = slmdata['slm']['t'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    mask = np.ones((1,64984))
    thresh = random.random()
    reselspvert = np.random.rand(1,64984)
    dummy_test(slm, mask, thresh, reselspvert)
    
def test_6():
    # generate random data, add reselpvert
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    mask = np.ones((1,k))
    thresh = random.random()
    reselspvert = np.random.rand(1,k)
    dummy_test(slm, mask, thresh, reselspvert)

def test_7():
    # generate random data, add reselpvert, special case slm['k']=2, l=1
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    slm['k'] = 2
    slm['df'] = random.randint(100, 10000)
    mask = np.ones((1,k))
    thresh = random.random()
    reselspvert = np.random.rand(1,k)
    dummy_test(slm, mask, thresh, reselspvert)
    
def test_8():
    # generate random data, add reselpvert, special case slm['k']=2, l>1,
    # l>1 is for instance when slm['t'] = np.random.rand(2,k) 
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    slm = {}
    slm['t'] = np.random.rand(2,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    slm['k'] = 2
    slm['df'] = random.randint(100, 10000)
    mask = np.ones((1,k))
    thresh = random.random()
    reselspvert = np.random.rand(1,k)
    dummy_test(slm, mask, thresh, reselspvert)

def test_9():
    # generate random data, special case slm['k']=3, l=1
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    slm['k'] = 3
    slm['df'] = random.randint(100, 10000)
    mask = np.ones((1,k))
    thresh = random.random()
    dummy_test(slm, mask, thresh,)

def test_10():
    # generate random data, add reselspvert, special case slm['k']=3, l=1
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    slm['k'] = 3
    slm['df'] = random.randint(100, 10000)
    mask = np.ones((1,k))
    thresh = random.random()
    reselspvert = np.random.rand(1,k)
    dummy_test(slm, mask, thresh, reselspvert)
    
def test_11():
    # generate random data, add reselspvert, special case slm['k']=3, l=2,
    # l=2 means that slm['t'] = np.random.rand(2,k) 
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    slm = {}
    slm['t'] = np.random.rand(2,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    slm['k'] = 3
    slm['df'] = random.randint(100, 10000)
    mask = np.ones((1,k))
    thresh = random.random()
    reselspvert = np.random.rand(1,k)
    dummy_test(slm, mask, thresh, reselspvert)

def test_12():
    # generate random data, add reselspvert, special case slm['k']=3, l=3,
    # l=3 means that slm['t'] = np.random.rand(3,k) 
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    slm = {}
    slm['t'] = np.random.rand(3,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    slm['k'] = 3
    slm['df'] = random.randint(100, 10000)
    mask = np.ones((1,k))
    thresh = random.random()
    reselspvert = np.random.rand(1,k)
    dummy_test(slm, mask, thresh, reselspvert)
    
def test_13():
    # generate random data, add reselpvert, random mask
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    mask = np.random.choice([0, 1], size=(1,k))
    thresh = random.random()
    reselspvert = np.random.rand(1,k)
    dummy_test(slm, mask, thresh, reselspvert)

def test_14():
    # generate random data, add reselspvert, special case slm['k']=3, l=2,
    # l=2 means that slm['t'] = np.random.rand(2,k) 
    # random mask
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    slm = {}
    slm['t'] = np.random.rand(2,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    slm['k'] = 3
    slm['df'] = random.randint(100, 10000)
    mask = np.random.choice([0, 1], size=(1,k))
    thresh = random.random()
    reselspvert = np.random.rand(1,k)
    dummy_test(slm, mask, thresh, reselspvert)

def test_15():
    # generate random data, small sized as 1000 points for slm['t'],
    # generate random edge, random reselspvert
    k = 1000
    m = 100
    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    mask = np.ones((1,k))
    thresh = random.random()
    reselspvert = np.random.rand(1,k)
    A = {}
    A['tri'] = np.random.randint(1,k, size=(m,3))    
    edg = py_SurfStatEdg(A)
    dummy_test(slm, mask, thresh, reselspvert, edg)
    
def test_16():
    # generate random data, add reselspvert, special case slm['k']=3, l=2,
    # l=2 means that slm['t'] = np.random.rand(2,k) 
    # random mask, generate random edge
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    slm = {}
    slm['t'] = np.random.rand(2,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    slm['k'] = 3
    slm['df'] = random.randint(100, 10000)
    mask = np.random.choice([0, 1], size=(1,k))
    thresh = random.random()
    reselspvert = np.random.rand(1,k)
    A = {}
    A['tri'] = np.random.randint(1,k, size=(m,3))
    edg = py_SurfStatEdg(A)
    dummy_test(slm, mask, thresh, reselspvert, edg)
    
def test_17():
    # generate random data, small sized as 1000 points for slm['t'],
    # generate random edge from random ['lat'], random reselspvert,
    # random mask
    k = 1000
    m = 100
    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    mask = np.random.choice([0, 1], size=(1,k))
    thresh = random.random()
    reselspvert = np.random.rand(1,k)
    A = {}
    A['lat'] =np.random.choice([0, 1], size=(10,10,10))
    edg = py_SurfStatEdg(A)
    dummy_test(slm, mask, thresh, reselspvert, edg)

def test_18():
    # generate random data, small sized as 1000 points for slm['t'],
    # extremely big threshold for n < 1 case in SurfStatPeakClus.py
    k = 1000
    m = 100
    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    mask = np.ones((1,k))
    thresh = k * m * 1000000
    dummy_test(slm, mask, thresh)



