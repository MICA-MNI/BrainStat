from brainstat.stats.LinMod import LinMod
from brainstat.stats.T import T
from brainstat.stats.term import Term
from brainstat.stats.P import P
import surfstat_wrap as sw
from scipy.io import loadmat
import numpy as np
import pytest

import os
import brainstat

sw.matlab_init_surfstat()


def dummy_test(slm, mask=None, clusthresh=0.001):

    try:
        # wrap matlab functions
        M_pval, M_peak, M_clus, M_clusid = sw.matlab_P(slm,
                                                               mask,
                                                               clusthresh)

    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # run python equivalent
    PY_pval, PY_peak, PY_clus, PY_clusid = P(slm,
                                                        mask,
                                                        clusthresh)

    # compare matlab-python outputs
    testout_P = []

    for key in M_pval:
        testout_P.append(np.allclose(M_pval[key], PY_pval[key],
                                      rtol=1e-05, equal_nan=True))
    for key in M_peak:
        testout_P.append(np.allclose(M_peak[key], PY_peak[key],
                                      rtol=1e-05, equal_nan=True))
    for key in M_clus:
        testout_P.append(np.allclose(M_clus[key], PY_clus[key],
                                      rtol=1e-05, equal_nan=True))
    testout_P.append(np.allclose(M_clusid, PY_clusid,
                              rtol=1e-05, equal_nan=True))


    assert (all(flag == True for (flag) in testout_P))


def test_01():
    # special case, v =1, l=1
    l = int(1)
    v = int(1)
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = np.random.rand(l,v)
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]
    slm['dfs'] =  np.array([[np.random.randint(1,10)]])
    P(slm)


def test_03():
    #special case, v=1, l>1, other input more randomized
    l = np.random.randint(1,10000)
    v = int(1)
    e = np.random.randint(1,10)
    k = 1
    d = np.random.randint(1111,2000)
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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

    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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

    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
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
    slmfile = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slmk3.mat')
    slmdata = loadmat(slmfile)
    slm = {}
    slm['t'] = slmdata['slm']['t'][0,0]
    slm['df'] = slmdata['slm']['df'][0,0]
    slm['k'] = slmdata['slm']['k'][0,0]
    slm['resl'] = slmdata['slm']['resl'][0,0]
    slm['tri'] = slmdata['slm']['tri'][0,0]


    dummy_test(slm)


def test_17():
    # load tutorial data (for n=10 subjects)
    fname = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'thickness.mat')
    f = loadmat(fname)
    SW = {}
    SW['tri'] = f['tri']
    SW['coord'] = f['coord']
    Y = f['T']
    AGE = np.array(f['AGE'])
    AGE = AGE.reshape(AGE.shape[1], 1)
    A = Term(AGE, 'AGE')
    M = 1 + A
    slm = LinMod(Y, M, SW)
    slm = T(slm, -1*AGE)
    dummy_test(slm)


def test_18():
    fname = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'thickness_slm.mat')
    f = loadmat(fname)
    slm = {}
    slm['X'] = f['slm']['X'][0,0]
    slm['df'] = f['slm']['df'][0,0][0,0]
    slm['coef'] = f['slm']['coef'][0,0]
    slm['SSE'] = f['slm']['SSE'][0,0]
    slm['tri'] = f['slm']['tri'][0,0]
    slm['resl'] = f['slm']['resl'][0,0]
    AGE = f['slm']['AGE'][0,0]
    slm = T(slm, -1*AGE)
    dummy_test(slm)


def test_19():
    fname = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'thickness_slm.mat')
    f = loadmat(fname)
    slm = {}
    slm['X'] = f['slm']['X'][0,0]
    slm['df'] = f['slm']['df'][0,0][0,0]
    slm['coef'] = f['slm']['coef'][0,0]
    slm['SSE'] = f['slm']['SSE'][0,0]
    slm['tri'] = f['slm']['tri'][0,0]
    slm['resl'] = f['slm']['resl'][0,0]
    AGE = f['slm']['AGE'][0,0]
    slm = T(slm, -1*AGE)
    mname = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'mask.mat')
    m = loadmat(mname)
    mask = m['mask'].astype(bool).flatten()
    dummy_test(slm, mask)


def test_20():
    fname = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'sofopofo1.mat')
    f = loadmat(fname)
    fT = f['sofie']['T'][0,0]
    params = f['sofie']['model'][0,0]
    colnames = ['1', 'ak', 'female', 'male', 'Affect', 'Control1',
                'Perspective', 'Presence', 'ink']
    M = Term(params, colnames)
    SW = {}
    SW['tri'] = f['sofie']['SW'][0,0]['tri'][0,0]
    SW['coord'] = f['sofie']['SW'][0,0]['coord'][0,0]
    slm = LinMod(fT, M, SW)
    contrast = np.array([[37], [41], [24], [37], [26], [28], [44], [26], [22],
                         [32], [34], [33], [35], [25], [22], [27], [22], [29],
                         [29], [24]])
    slm = T(slm, contrast)
    dummy_test(slm)


def test_21():
    fname = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'sofopofo1_slm.mat')
    f = loadmat(fname)
    slm = {}
    slm['X'] = f['slm']['X'][0,0]
    slm['df'] = f['slm']['df'][0,0][0,0]
    slm['coef'] = f['slm']['coef'][0,0]
    slm['SSE'] = f['slm']['SSE'][0,0]
    slm['tri'] = f['slm']['tri'][0,0]
    slm['resl'] = f['slm']['resl'][0,0]
    contrast = np.random.randint(20,50, size=(slm['X'].shape[0],1))
    slm = T(slm, contrast)
    mask = np.random.choice([0, 1], size=(slm['t'].shape[1]))
    mask = mask.astype(bool).flatten()
    dummy_test(slm, mask)
