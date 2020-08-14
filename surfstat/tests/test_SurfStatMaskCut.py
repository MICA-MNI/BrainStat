import sys
sys.path.append("/data/p_02323/BrainStat/surfstat/")
import surfstat_wrap as sw
from SurfStatMaskCut import *
import numpy as np
import random
import pytest

def dummy_test(surf):

    try:
        # wrap matlab functions
        M_mask = sw.matlab_SurfStatMaskCut(surf)
    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # run python equivalent
    P_mask = py_SurfStatMaskCut(surf)
    
    # compare matlab-python outputs
    assert np.allclose(M_mask, P_mask, rtol=1e-05, equal_nan=True)


sw.matlab_init_surfstat()

#%Mask that excludes the inter-hemisphere cut.
#%
#% Usage: mask = SurfStatMaskCut( surf );
#%
#% surf.coord   = 3 x v matrix of surface coordinates, v=#vertices.
#%
#% mask         = 1 x v vector, 1=inside, 0=outside, v=#vertices.
#%
#% It looks in -50<y<50 and -20<z<40, and mask vertices where |x|>thresh,
#% where thresh = 1.5 x arg max of a histogram of |x|. 

def test_1():
    v = np.random.randint(10,100)
    surf = {}
    surf['coord'] = np.random.rand(3,v)
    dummy_test(surf)
    
def test_2():
    v = np.random.randint(100,1000)
    surf = {}
    surf['coord'] = np.random.rand(3,v)
    dummy_test(surf)
        
def test_3():
    v = np.random.randint(1000,3000)
    surf = {}
    surf['coord'] = np.random.rand(3,v)
    dummy_test(surf)        

def test_4():
    v = np.random.randint(3000,10000)
    surf = {}
    surf['coord'] = np.random.rand(3,v)
    dummy_test(surf)        

        
def test_5():
    v = np.random.randint(10,100)
    surf = {}
    surf['coord'] = np.random.uniform(low=-3, high=3, size=(3,v))
    dummy_test(surf)
    
def test_6():
    v = np.random.randint(100,1000)
    surf = {}
    surf['coord'] = np.random.uniform(low=-3, high=3, size=(3,v))
    dummy_test(surf)

def test_7():
    v = np.random.randint(1000,3000)
    surf = {}
    surf['coord'] = np.random.uniform(low=-3, high=3, size=(3,v))
    dummy_test(surf)
    
def test_8():
    v = np.random.randint(3000,10000)
    surf = {}
    surf['coord'] = np.random.uniform(low=-3, high=3, size=(3,v))
    dummy_test(surf)
    
    
def test_9():
    v = np.random.randint(10,100)
    surf = {}
    surf['coord'] = np.random.uniform(low=-70, high=70, size=(3,v))
    dummy_test(surf)
    
def test_10():
    v = np.random.randint(100,1000)
    surf = {}
    surf['coord'] = np.random.uniform(low=-70, high=70, size=(3,v))
    dummy_test(surf)

def test_11():
    v = np.random.randint(1000,3000)
    surf = {}
    surf['coord'] = np.random.uniform(low=-70, high=70, size=(3,v))
    dummy_test(surf)
    
def test_12():
    v = np.random.randint(3000,10000)
    surf = {}
    surf['coord'] = np.random.uniform(low=-70, high=70, size=(3,v))
    dummy_test(surf)
    
    
def test_13():
    v = np.random.randint(10,100)
    surf = {}
    surf['coord'] = np.random.randint(low=-3, high=3, size=(3,v))
    dummy_test(surf)    
    
def test_14():
    v = np.random.randint(100,1000)
    surf = {}
    surf['coord'] = np.random.randint(low=-3, high=3, size=(3,v))
    dummy_test(surf)    

def test_15():
    v = np.random.randint(1000,3000)
    surf = {}
    surf['coord'] = np.random.randint(low=-3, high=3, size=(3,v))
    dummy_test(surf)    

def test_16():
    v = np.random.randint(3000,10000)
    surf = {}
    surf['coord'] = np.random.randint(low=-3, high=3, size=(3,v))
    dummy_test(surf)    
    
    
def test_17():
    v = np.random.randint(10,100)
    surf = {}
    surf['coord'] = np.random.randint(low=-70, high=70, size=(3,v))
    dummy_test(surf)    
    
def test_18():
    v = np.random.randint(100,1000)
    surf = {}
    surf['coord'] = np.random.randint(low=-70, high=70, size=(3,v))
    dummy_test(surf)    

def test_19():
    v = np.random.randint(1000,3000)
    surf = {}
    surf['coord'] = np.random.randint(low=-70, high=70, size=(3,v))
    dummy_test(surf)    

def test_20():
    v = np.random.randint(3000,10000)
    surf = {}
    surf['coord'] = np.random.randint(low=-70, high=70, size=(3,v))
    dummy_test(surf)    
        
        
def test_21():
    v = np.random.randint(10000,30000)
    surf = {}
    surf['coord'] = np.random.uniform(low=-50, high=50, size=(3,v))
    dummy_test(surf)
    
def test_22():
    v = np.random.randint(30000, 100000)
    surf = {}
    surf['coord'] = np.random.uniform(low=-1000, high=1000, size=(3,v))
    dummy_test(surf)
    
    
def test_23():
    v = np.random.randint(30000, 100000)
    surf = {}
    surf['coord'] = np.random.uniform(low=-3, high=3, size=(3,v))
    dummy_test(surf)
    
def test_24():
    v = np.random.randint(30000, 100000)
    surf = {}
    surf['coord'] = np.random.uniform(low=-3, high=3, size=(3,v))
    dummy_test(surf)
    

