import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats.multiple_comparisons import _fdr


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
    if 'dfs' in idic.keys():
        slm['dfs'] = idic['dfs']

    if 'resl' in idic.keys():
        slm['resl'] = idic['resl']

    if 'tri' in idic.keys():
        slm['tri']    = idic['tri']

    if 'lat' in idic.keys():
        slm['tri']    = idic['tri']

    if 'mask' in idic.keys():
        mask = idic['mask']
    else:
        mask = None

    # non-sense input for _fdr, but potentially produced by SurfStatLinMod
    if 'du' in idic.keys():
        slm['du'] = idic['du']

    if 'c' in idic.keys():
        slm['c']    = idic['c']

    if 'ef' in idic.keys():
        slm['ef']    = idic['ef']

    if 'sd' in idic.keys():
        slm['sd']    = idic['sd']

    if 'SSE' in idic.keys():
        slm['SSE']    = idic['SSE']

    # run _fdr
    outdic = _fdr(slm, mask)

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
    # real-data testing with only mandatory input slm['t'], slm['df'], slm['k']
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] : np array, shape (1, 1), uint16
    # ['k'] : np array, shape (1, 1), uint8
    infile  = datadir('statq_01_IN.pkl')
    expfile = datadir('statq_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02():
    # test data, slm['t'] and slm['df'] 2D arrays, slm['k'] int
    # ['t'] : np array, shape (1, 9850), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    infile  = datadir('statq_02_IN.pkl')
    expfile = datadir('statq_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03():
    # similar to test_02, shapes/values of slm['t'] and slm['df'] manipulated
    # ['t'] :  np array, shape (1, 2139), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] :  int
    infile  = datadir('statq_03_IN.pkl')
    expfile = datadir('statq_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04():
    # similar to test_02 + optional input ['mask']
    # ['t'] : np array, shape (1, 2475), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['mask'] : np array, shape (2475,), bool
    infile  = datadir('statq_04_IN.pkl')
    expfile = datadir('statq_04_OUT.pkl')
    dummy_test(infile, expfile)


def test_05():
    # similar to test_02 + optional input slm['dfs']
    # ['t'] : np array, shape (1, 1998), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['dfs'] : np array, shape (1, 1998), int64
    infile  = datadir('statq_05_IN.pkl')
    expfile = datadir('statq_05_OUT.pkl')
    dummy_test(infile, expfile)


def test_06():
    # similar to test_02 + optional inputs slm['dfs'] and ['mask']
    # ['t'] : np array, shape (1, 3328), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['dfs'] : np array, shape (1, 3328), int64
    # ['mask'] : np array, shape (3328,), bool
    infile  = datadir('statq_06_IN.pkl')
    expfile = datadir('statq_06_OUT.pkl')
    dummy_test(infile, expfile)


def test_07():
    # similar to test_02 + optional inputs slm['dfs'], ['mask'] and ['tri']
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
    # similar to test_02 + optional inputs slm['dfs'], slm['tri'] and slm['resl']
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
    # similar to test_08 + values/shapes of input params changed +
    # additional input slm['du'] (non-sense for _fdr)
    # ['t'] : np array, shape (1, 4397), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['tri'] : np array, shape (2734, 3), int64
    # ['resl'] : np array, shape (8199, 1), float64
    # ['dfs'] : np array, shape (1, 4397), float64
    # ['du'] : int
    infile  = datadir('statq_09_IN.pkl')
    expfile = datadir('statq_09_OUT.pkl')
    dummy_test(infile, expfile)


def test_10():
    # similar to test_08 + + values/shapes of input params changed + additional
    # input slm['du'], slm['c'], slm['ef'], and slm['sd'] (non-sense for _fdr)
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
    # similar to test_08 + additional input ['c'], ['ef'], ['sd'], ['X'],
    # and ['coef'], ['SSE'] (non-sense for _fdr)
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
    # similar to test_11 + optional input ['mask'] + ['df'] dtype changed
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
    # similar to test_11, ['t']-values shuffled
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
    # similar to test_11, ['t']-values shuffled + optional ['mask']
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


