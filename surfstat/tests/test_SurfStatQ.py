import sys
sys.path.append("python")
from SurfStatQ import *
import surfstat_wrap as sw
import numpy as np
import pytest
from scipy.io import loadmat
import random

def dummy_test(slm, mask=None):

    try:
        # wrap matlab functions
        M_q_val = sw.matlab_SurfStatQ(slm, mask)
   
    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # run python equivalent
    P_q_val = py_SurfStatQ(slm, mask)
    
    # compare matlab-python outputs
    testout_SurfStatQ = []

    for key in M_q_val :
        testout_SurfStatQ.append(np.allclose(M_q_val[key], P_q_val[key], 
                                      rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout_SurfStatQ)

sw.matlab_init_surfstat()

def test_1():
    # data from Sofie, only slm['t'], slm['df'], slm['k'] --> mandatory input
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = slmdata['slm']['t'][0,0]
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    
    dummy_test(slm)
    
def test_2():
    # randomize slm['t'] and slm['df'], slm['k']=1 
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    
    slm = {}
    slm['t'] = np.random.rand(1, k) 
    slm['df'] =  np.array([[m]])
    slm['k'] = 1
    dummy_test(slm)    
    
def test_3():
    # randomize slm['t'] and slm['df'], slm['k']
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(50, 100)

    slm = {}
    slm['t'] = np.random.rand(1, k) 
    slm['df'] =  np.array([[m]])
    slm['k'] = n
    dummy_test(slm)

def test_4():
    # randomize slm['t'] and slm['df'], slm['k'], and a random mask (type bool)
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(50, 100)

    slm = {}
    slm['t'] = np.random.rand(1, k) 
    slm['df'] =  np.array([[m]])
    slm['k'] = n
    mask = np.random.choice([0, 1], size=(1,k))
    mask = mask.astype(bool)
    dummy_test(slm, mask)
   
def test_5():
    # randomize slm['t'] and slm['df'], slm['dfs'], slm['k']=1 
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    
    slm = {}
    slm['t'] = np.random.rand(1, k) 
    slm['df'] =  np.array([[m]])
    slm['k'] = 1
    slm['dfs'] = np.random.choice([1,k-1], size=(1,k))
    dummy_test(slm)
    
def test_6():
    # randomize slm['t'] and slm['df'], slm['k'], slm['dfs'] and a random mask (type bool)
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(50, 100)

    slm = {}
    slm['t'] = np.random.rand(1, k) 
    slm['df'] =  np.array([[m]])
    slm['k'] = n
    mask = np.random.choice([0, 1], size=(1,k))
    mask = mask.astype(bool)
    slm['dfs'] = np.random.choice([1,k-1], size=(1,k))
    dummy_test(slm, mask)
    
def test_7():
    # randomize slm['t'], slm['df'], slm['k'], slm['tri'], slm['dfs'], mask 
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(50, 100)
    
    slm = {}
    slm['t'] = np.random.rand(1, k) 
    slm['df'] =  np.array([[m]])
    slm['k'] = n
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    slm['dfs'] = np.random.choice([1,k-1], size=(1,k))
    mask = np.random.choice([0, 1], size=(1,k))
    mask = mask.astype(bool)
    dummy_test(slm, mask)
    
def test_8():
    # random slm['t'], slm['df'], slm['k'], slm['tri'], slm['resl'], slm['dfs']
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(1,10)

    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['df'] =  np.array([[m]])
    slm['k'] = 5
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    slm['resl'] = np.random.rand(k,1)
    slm['dfs'] = np.random.randint(1,10, (1,k))
    dummy_test(slm)    
    
def test_9():
    # random slm['t'], slm['df'], slm['tri'], slm['resl'],
    # special input case: slm['dfs'] and slm['du']
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(1,10)

    slm = {}
    slm['t'] = np.random.rand(1,k) 
    slm['df'] =  np.array([[m]])
    slm['k'] = 1 
    slm['du'] = n
    slm['tri'] = np.random.randint(1,k, size=(m,3))
    slm['resl'] = np.random.rand(k*2,1)
    slm['dfs'] = np.ones((1, k))
    dummy_test(slm) 




