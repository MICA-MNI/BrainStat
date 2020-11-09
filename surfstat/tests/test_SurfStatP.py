import sys
sys.path.append("python")
from SurfStatP import *
import surfstat_wrap as sw
from scipy.io import loadmat
import numpy as np
import pytest

sw.matlab_init_surfstat()


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


def test_01():
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


def test_02():
    # special case, v=1, l>1
    l = np.random.randint(1,10000)
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
    py_SurfStatP(slm)


def test_03():
    #special case, v=1, l>1, other input more randomized
    l = np.random.randint(1,10000)
    v = int(1)
    e = np.random.randint(1,10)
    k = 1
    d = np.random.randint(1111,2000)
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = np.random.rand(l,v)
    slm['df'] = np.array([[d]])
    slm['k'] = k
    slm['resl'] = np.random.rand(e,k)
    slm['tri'] = slmdata['slm']['tri'][0,0]
    slm['dfs'] =  np.array([[np.random.randint(1,10)]])
    dummy_test(slm)


def test_04():
    # v >1 and clusthresh < 1 (default clusthresh)
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = slmdata['slm']['t'][0,0]
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    dummy_test(slm,  mask=None, clusthresh=0.001)


def test_05():
    # v >1 and clusthresh < 1 (default clusthresh)
    l = 1
    v = 64984
    e = 194940
    k = 1
    d = 1111
    slm = {}
    slm['t'] = np.random.uniform(-5,5, (l,v))
    slm['df'] = np.array([[d]])
    slm['k'] = k
    slm['resl'] = np.random.rand(e,k)
    slm['tri'] = np.random.randint(low=1, high=64984+1, size=(129960, 3))
    dummy_test(slm,  mask=None, clusthresh=0.001)


def test_06():
    # v >1 and clusthresh < 1 (default clusthresh)
    l = 1
    v = 64984
    e = 194940
    k = 1
    d = np.random.randint(1111, 2000)
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = np.random.uniform(-5,5, (l,v))
    slm['df'] = np.array([[d]])
    slm['k'] = k
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    dummy_test(slm,  mask=None, clusthresh=0.001)


def test_07():
    # special case np.max(slm['t'][0, mask.flatten()]) < thresh
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = np.random.uniform(low=-4, high=0, size=(1,64984))
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    dummy_test(slm)


def test_08():
    # special case np.max(slm['t'][0, mask.flatten()]) < thresh
    # make slm['df'] a random integer
    d = np.random.randint(1111, 2000)
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = np.random.uniform(low=-4, high=0, size=(1,64984))
    slm['df'] = np.array([[d]])
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    dummy_test(slm)


def test_09():
    # special case case np.max(slm['t'][0, mask.flatten()]) > thresh
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = np.random.uniform(low=0, high=4, size=(1,64984))
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    dummy_test(slm)


def test_10():
    # data from Sofie
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = slmdata['slm']['t'][0,0]
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    dummy_test(slm)


def test_11():
    # data from Sofie + a random mask
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = slmdata['slm']['t'][0,0]
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]

    v = np.shape(slmdata['slm']['t'][0,0])[1]
    Amask = np.random.choice([0, 1], size=(v))
    Amask = np.array(Amask, dtype=bool)

    dummy_test(slm, mask=Amask)


def test_12():
    # data from Sofie + clusthresh is a random value
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)

    slm = {}

    slm['t'] = slmdata['slm']['t'][0,0]
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]

    Amask = np.ones((slm['t'].shape[1]))
    Amask = np.array(Amask, dtype=bool)
    Aclusthresh = 0.3
    dummy_test(slm, Amask, Aclusthresh)


def test_13():
    # randomize Sofie's data a little bit
    v = int(64984)
    y = int(194940)

    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)

    slm = {}

    slm['t'] = np.random.rand(1,v)
    slm['df'] = np.array([1111])
    slm['k'] = 1
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]

    dummy_test(slm)


def test_14():
    # data from Sofie, slm['t'] is array of shape (1,1)
    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)

    slm = {}

    slm['t'] = np.array([[-0.1718374541922737]])
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = 1
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]

    dummy_test(slm)


def test_15():
    # data from Sofie + add a random slm['dfs']
    v = int(64984)

    slmfile = './tests/data/slm.mat'
    slmdata = loadmat(slmfile)

    slm = {}

    slm['t'] = slmdata['slm']['t'][0,0]
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    slm['dfs'] = np.random.randint(1,10, (1,v))

    dummy_test(slm)


def test_16():
    # data from Reinder, slm.k = 3
    slmfile = './tests/data/slmk3.mat'
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = slmdata['slm']['t'][0,0]
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]


    dummy_test(slm)
