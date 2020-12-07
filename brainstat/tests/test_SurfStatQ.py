from brainstat.stats.SurfStatQ import SurfStatQ
from brainstat.stats.term import Term
from brainstat.stats.SurfStatEdg import SurfStatEdg
from brainstat.stats.SurfStatLinMod import SurfStatLinMod
from brainstat.stats.SurfStatT import SurfStatT

import surfstat_wrap as sw
import numpy as np
import pytest
from scipy.io import loadmat
import random

import os
import brainstat

sw.matlab_init_surfstat()


def dummy_test(slm, mask=None):

    try:
        # wrap matlab functions
        M_q_val = sw.matlab_Q(slm, mask)

    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # run python equivalent
    P_q_val = SurfStatQ(slm, mask)

    # compare matlab-python outputs
    testout_Q = []

    for key in M_q_val:
        testout_Q.append(np.allclose(np.squeeze(M_q_val[key]),
                                     np.squeeze(P_q_val[key]),
                                     rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout_Q)


sw.matlab_init_surfstat()


def test_01():
    # data from Sofie, only slm['t'], slm['df'], slm['k'] --> mandatory input
    slmdata = loadmat(os.path.dirname(brainstat.__file__) +
                      os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'slm.mat')
    slm = {}
    slm['t'] = slmdata['slm']['t'][0, 0]
    slm['df'] = slmdata['slm']['df'][0, 0]
    slm['k'] = slmdata['slm']['k'][0, 0]

    dummy_test(slm)


def test_02():
    # randomize slm['t'] and slm['df'], slm['k']=1
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)

    slm = {}
    slm['t'] = np.random.rand(1, k)
    slm['df'] = np.array([[m]])
    slm['k'] = 1
    dummy_test(slm)


def test_03():
    # randomize slm['t'] and slm['df'], slm['k']
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(1, 3)

    slm = {}
    slm['t'] = np.random.rand(1, k)
    slm['df'] = np.array([[m]])
    slm['k'] = n
    dummy_test(slm)


def test_04():
    # randomize slm['t'] and slm['df'], slm['k'], and a random mask (type bool)
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(1, 3)

    slm = {}
    slm['t'] = np.random.rand(1, k)
    slm['df'] = np.array([[m]])
    slm['k'] = n
    mask = np.random.choice([0, 1], size=(k))
    mask = mask.astype(bool)
    dummy_test(slm, mask)


def test_05():
    # randomize slm['t'] and slm['df'], slm['dfs'], slm['k']=1
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)

    slm = {}
    slm['t'] = np.random.rand(1, k)
    slm['df'] = np.array([[m]])
    slm['k'] = 1
    slm['dfs'] = np.random.choice([1, k-1], size=(1, k))
    dummy_test(slm)


def test_06():
    # randomize slm['t'] and slm['df'], slm['k'], slm['dfs'] and a random mask (type bool)
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(1, 3)

    slm = {}
    slm['t'] = np.random.rand(1, k)
    slm['df'] = np.array([[m]])
    slm['k'] = n
    mask = np.random.choice([0, 1], size=(k))
    mask = mask.astype(bool)
    slm['dfs'] = np.random.choice([1, k-1], size=(1, k))
    dummy_test(slm, mask)


def test_07():
    # randomize slm['t'], slm['df'], slm['k'], slm['tri'], slm['dfs'], mask
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(1, 3)

    slm = {}
    slm['t'] = np.random.rand(1, k)
    slm['df'] = np.array([[m]])
    slm['k'] = n
    slm['tri'] = np.random.randint(1, k, size=(m, 3))
    slm['dfs'] = np.random.choice([1, k-1], size=(1, k))
    mask = np.random.choice([0, 1], size=(k))
    mask = mask.astype(bool)
    dummy_test(slm, mask)


def test_08():
    # random slm['t'], slm['df'], slm['k'], slm['tri'], slm['resl'], slm['dfs']
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(1, 10)

    slm = {}
    slm['t'] = np.random.rand(1, k)
    slm['df'] = np.array([[m]])
    slm['k'] = 5
    slm['tri'] = np.random.randint(1, k, size=(m, 3))
    slm['resl'] = np.random.rand(k, 1)
    slm['dfs'] = np.random.randint(1, 10, (1, k))
    dummy_test(slm)


def test_09():
    # random slm['t'], slm['df'], slm['tri'], slm['resl'],
    # special input case: slm['dfs'] and slm['du']
    k = random.randint(1000, 10000)
    m = random.randint(1000, 10000)
    n = random.randint(1, 10)

    slm = {}
    slm['t'] = np.random.rand(1, k)
    slm['df'] = np.array([[m]])
    slm['k'] = 1
    slm['du'] = n
    slm['tri'] = np.random.randint(1, k, size=(m, 3))
    edg = SurfStatEdg(slm)
    slm['resl'] = np.random.rand(edg.shape[0], 1)
    slm['dfs'] = np.ones((1, k))
    dummy_test(slm)


def test_10():
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
    slm = SurfStatLinMod(Y, M, SW)
    slm = SurfStatT(slm, -1*AGE)
    dummy_test(slm)


def test_11():
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
    slm = SurfStatLinMod(Y, M, SW)
    slm = SurfStatT(slm, -1*AGE)
    dummy_test(slm)


def test_12():
    fname = (os.path.dirname(brainstat.__file__) +
             os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'thickness_slm.mat')
    f = loadmat(fname)
    slm = {}
    slm['X'] = f['slm']['X'][0, 0]
    slm['df'] = f['slm']['df'][0, 0][0, 0]
    slm['coef'] = f['slm']['coef'][0, 0]
    slm['SSE'] = f['slm']['SSE'][0, 0]
    slm['tri'] = f['slm']['tri'][0, 0]
    slm['resl'] = f['slm']['resl'][0, 0]
    AGE = f['slm']['AGE'][0, 0]
    slm = SurfStatT(slm, -1*AGE)

    mname = (os.path.dirname(brainstat.__file__) +
             os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'mask.mat')
    m = loadmat(mname)
    mask = m['mask'].astype(bool).flatten()

    dummy_test(slm, mask)


def test_13():
    fname = (os.path.dirname(brainstat.__file__) +
             os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'sofopofo1.mat')
    f = loadmat(fname)
    fT = f['sofie']['T'][0, 0]

    params = f['sofie']['model'][0, 0]
    colnames = ['1', 'ak', 'female', 'male', 'Affect', 'Control1',
                'Perspective', 'Presence', 'ink']
    M = Term(params, colnames)
    SW = {}
    SW['tri'] = f['sofie']['SW'][0, 0]['tri'][0, 0]
    SW['coord'] = f['sofie']['SW'][0, 0]['coord'][0, 0]
    slm = SurfStatLinMod(fT, M, SW)
    contrast = np.random.randint(20, 50, size=(slm['X'].shape[0], 1))
    slm = SurfStatT(slm, contrast)
    dummy_test(slm)


def test_14():
    fname = (os.path.dirname(brainstat.__file__) +
             os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'sofopofo1_slm.mat')
    f = loadmat(fname)
    slm = {}
    slm['X'] = f['slm']['X'][0, 0]
    slm['df'] = f['slm']['df'][0, 0][0, 0]
    slm['coef'] = f['slm']['coef'][0, 0]
    slm['SSE'] = f['slm']['SSE'][0, 0]
    slm['tri'] = f['slm']['tri'][0, 0]
    slm['resl'] = f['slm']['resl'][0, 0]
    contrast = np.random.randint(20, 50, size=(slm['X'].shape[0], 1))
    slm = SurfStatT(slm, contrast)
    mask = np.random.choice([0, 1], size=(slm['t'].shape[1]))
    mask = mask.astype(bool).flatten()
    dummy_test(slm, mask)
