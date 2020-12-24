import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatQ


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    slm = {}
    slm['t']  = idic['t']
    slm['df'] = idic['df']
    slm['k']  = idic['k']

    # check other potential keys in input
    if 'tri' in idic.keys():
        slm['tri']    = idic['tri']

    if 'resl' in idic.keys():
        slm['resl'] = idic['resl']

    if 'dfs' in idic.keys():
        slm['dfs'] = idic['dfs']

    if 'du' in idic.keys():
        slm['du'] = idic['du']

    if 'c' in idic.keys():
        slm['c']    = idic['c']

    if 'k' in idic.keys():
        slm['k']    = idic['k']

    if 'ef' in idic.keys():
        slm['ef']    = idic['ef']

    if 'sd' in idic.keys():
        slm['sd']    = idic['sd']

    if 't' in idic.keys():
        slm['t']    = idic['t']

    if 'SSE' in idic.keys():
        slm['SSE']    = idic['SSE']

    if 'mask' in idic.keys():
        mask = idic['mask']
    else:
        mask = None


    # run SurfStatQ
    outdic = SurfStatQ(slm, mask)

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in outdic.keys():
        comp = np.allclose(outdic[key], expdic[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] : np array, shape (1, 1), uint16
    # ['k'] : np array, shape (1, 1), uint8
    infile  = datadir('statq_01_IN.pkl')
    expfile = datadir('statq_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02():
    # ['t'] : np array, shape (1, 9850), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    infile  = datadir('statq_02_IN.pkl')
    expfile = datadir('statq_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03():
    # ['t'] :  np array, shape (1, 2139), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] :  int
    infile  = datadir('statq_03_IN.pkl')
    expfile = datadir('statq_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04():
    # ['t'] : np array, shape (1, 2475), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['mask'] : np array, shape (2475,), bool
    infile  = datadir('statq_04_IN.pkl')
    expfile = datadir('statq_04_OUT.pkl')
    dummy_test(infile, expfile)


def test_05():
    # ['t'] : np array, shape (1, 1998), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['dfs'] : np array, shape (1, 1998), int64
    infile  = datadir('statq_05_IN.pkl')
    expfile = datadir('statq_05_OUT.pkl')
    dummy_test(infile, expfile)


def test_06():
    # ['t'] : np array, shape (1, 3328), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['dfs'] : np array, shape (1, 3328), int64
    # ['mask'] : np array, shape (3328,), bool
    infile  = datadir('statq_06_IN.pkl')
    expfile = datadir('statq_06_OUT.pkl')
    dummy_test(infile, expfile)


def test_07():
    # ['t'] : np array, shape (1, 9512), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['dfs'] : np array, shape (1, 9512), int64
    # ['mask'] : np array, shape (9512,), bool
    # ['tri'] : np array, shape (1724, 3), int64
    infile  = datadir('statq_07_IN.pkl')
    expfile = datadir('statq_07_OUT.pkl')
    dummy_test(infile, expfile)


def test_08():
    # ['t'] : np array, shape (1, 1520), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['dfs'] : np array, shape (1, 1520), int64
    # ['tri'] : np array, shape (4948, 3), int64
    # ['resl'] : np array, shape (1520, 1), float64
    infile  = datadir('statq_08_IN.pkl')
    expfile = datadir('statq_08_OUT.pkl')
    dummy_test(infile, expfile)


def test_09():
    # ['t'] : np array, shape (1, 4397), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['du'] : int
    # ['tri'] : np array, shape (2734, 3), int64
    # ['resl'] : np array, shape (8199, 1), float64
    # ['dfsl'] : np array, shape (1, 4397), float64
    infile  = datadir('statq_09_IN.pkl')
    expfile = datadir('statq_09_OUT.pkl')
    dummy_test(infile, expfile)


def test_10():
    # ['t'] : np array, shape (1, 20484), float64
    # ['df'] : int64
    # ['k'] : int
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['c'] : np array, shape (1, 2), float64
    # ['ef'] : np array, shape (1, 20484), float64
    # ['sd'] : np array, shape (1, 20484), float64
    infile  = datadir('statq_10_IN.pkl')
    expfile = datadir('statq_10_OUT.pkl')
    dummy_test(infile, expfile)


def test_11():
    # ['t'] : np array, shape (1, 20484), float64
    # ['df'] : int64
    # ['k'] : int
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['c'] : np array, shape (1, 2), float64
    # ['ef'] : np array, shape (1, 20484), float64
    # ['sd'] : np array, shape (1, 20484), float64
    # ['X'] : np array, shape (10, 2), float64
    # ['coef'] : np array, shape (2, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    infile  = datadir('statq_11_IN.pkl')
    expfile = datadir('statq_11_OUT.pkl')
    dummy_test(infile, expfile)


def test_12():
    # ['t'] : np array, shape (1, 20484), float64
    # ['df'] : uint8
    # ['k'] : int
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['c'] : np array, shape (1, 2), float64
    # ['ef'] : np array, shape (1, 20484), float64
    # ['sd'] : np array, shape (1, 20484), float64
    # ['X'] : np array, shape (10, 2), uint8
    # ['coef'] : np array, shape (2, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['mask'] : np array, shape (20484,), bool
    infile  = datadir('statq_12_IN.pkl')
    expfile = datadir('statq_12_OUT.pkl')
    dummy_test(infile, expfile)


def test_13():
    # ['t'] : np array, shape (1, 20484), float64
    # ['df'] : int64
    # ['k'] : int
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['c'] : np array, shape (1, 9), float64
    # ['ef'] : np array, shape (1, 20484), float64
    # ['sd'] : np array, shape (1, 20484), float64
    # ['X'] : np array, shape (20, 9), uint16
    # ['coef'] : np array, shape (9, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    infile  = datadir('statq_13_IN.pkl')
    expfile = datadir('statq_13_OUT.pkl')
    dummy_test(infile, expfile)


def test_14():
    # ['t'] : np array, shape (1, 20484), float64
    # ['df'] : uint8
    # ['k'] : int
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['c'] : np array, shape (1, 9), float64
    # ['ef'] : np array, shape (1, 20484), float64
    # ['sd'] : np array, shape (1, 20484), float64
    # ['X'] : np array, shape (20, 9), uint16
    # ['coef'] : np array, shape (9, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['mask'] : np array, shape (20484,), bool
    infile  = datadir('statq_14_IN.pkl')
    expfile = datadir('statq_14_OUT.pkl')
    dummy_test(infile, expfile)


