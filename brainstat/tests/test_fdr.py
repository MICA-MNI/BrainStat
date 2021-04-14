import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats._multiple_comparisons import fdr
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import Term


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    slm = SLM(Term(1), Term(1))
    for key in idic.keys():
        setattr(slm, key, idic[key])

    # run fdr
    Q = fdr(slm)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    assert np.allclose(Q, expdic["Q"])


def test_01():
    # random data shape matching a real-data set
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] : int
    # ['k'] : int
    infile = datadir("xstatq_01_IN.pkl")
    expfile = datadir("xstatq_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    # random data
    # ['t'] : np array, shape (1, 9850), float64
    # ['df'] : int
    # ['k'] : int
    infile = datadir("xstatq_02_IN.pkl")
    expfile = datadir("xstatq_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    # similar to test_02, shapes/values of slm['t'] and slm['df'] manipulated
    # ['t'] :  np array, shape (1, 2139), float64
    # ['df'] : int
    # ['k'] :  int
    infile = datadir("xstatq_03_IN.pkl")
    expfile = datadir("xstatq_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    # similar to test_02 + optional input ['mask']
    # ['t'] : np array, shape (1, 2475), float64
    # ['df'] : int
    # ['k'] : int
    # ['mask'] : np array, shape (2475,), bool
    infile = datadir("xstatq_04_IN.pkl")
    expfile = datadir("xstatq_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    # similar to test_02 + optional input slm['dfs']
    # ['t'] : np array, shape (1, 1998), float64
    # ['df'] : int
    # ['k'] : int
    # ['dfs'] : np array, shape (1, 1998), int64
    infile = datadir("xstatq_05_IN.pkl")
    expfile = datadir("xstatq_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    # similar to test_02 + optional inputs slm['dfs'] and ['mask']
    # ['t'] : np array, shape (1, 3328), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['dfs'] : np array, shape (1, 3328), int64
    # ['mask'] : np array, shape (3328,), bool
    infile = datadir("xstatq_06_IN.pkl")
    expfile = datadir("xstatq_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    # similar to test_02 + optional inputs slm['dfs'], ['mask'] and ['tri']
    # ['t'] : np array, shape (1, 9512), float64
    # ['df'] : int
    # ['k'] : int
    # ['dfs'] : np array, shape (1, 9512), int64
    # ['mask'] : np array, shape (9512,), bool
    # ['tri'] : np array, shape (1724, 3), int64
    infile = datadir("xstatq_07_IN.pkl")
    expfile = datadir("xstatq_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    # similar to test_02 + optional inputs slm['dfs'], slm['tri'] and slm['resl']
    # ['t'] : np array, shape (1, 1520), float64
    # ['df'] : int
    # ['k'] : int
    # ['dfs'] : np array, shape (1, 1520), int64
    # ['tri'] : np array, shape (4948, 3), int64
    # ['resl'] : np array, shape (1520, 1), float64
    infile = datadir("xstatq_08_IN.pkl")
    expfile = datadir("xstatq_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    # similar to test_08 + values/shapes of input params changed +
    # additional input slm['du'] (non-sense for _fdr)
    # ['t'] : np array, shape (1, 4397), float64
    # ['df'] : int
    # ['k'] : int
    # ['tri'] : np array, shape (2734, 3), int64
    # ['resl'] : np array, shape (8199, 1), float64
    # ['dfs'] : np array, shape (1, 4397), float64
    # ['du'] : int
    infile = datadir("xstatq_09_IN.pkl")
    expfile = datadir("xstatq_09_OUT.pkl")
    dummy_test(infile, expfile)


def test_10():
    # similar to test_08 + + values/shapes of input params changed + additional
    # input slm['du'], slm['c'], slm['ef'], and slm['sd'] (non-sense for _fdr)
    # ['t'] : np array, shape (1, 20484), float64
    # ['df'] : int
    # ['k'] : int
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['c'] : np array, shape (1, 2), float64
    # ['ef'] : np array, shape (1, 20484), float64
    # ['sd'] : np array, shape (1, 20484), float64
    infile = datadir("xstatq_10_IN.pkl")
    expfile = datadir("xstatq_10_OUT.pkl")
    dummy_test(infile, expfile)


def test_11():
    # similar to test_08 + additional input ['c'], ['ef'], ['sd'], ['X'],
    # and ['coef'], ['SSE'] (non-sense for _fdr)
    # ['t'] : np array, shape (1, 20484), float64
    # ['df'] : int
    # ['k'] : int
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['c'] : np array, shape (1, 2), float64
    # ['ef'] : np array, shape (1, 20484), float64
    # ['sd'] : np array, shape (1, 20484), float64
    # ['X'] : np array, shape (10, 2), float64
    # ['coef'] : np array, shape (2, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    infile = datadir("xstatq_11_IN.pkl")
    expfile = datadir("xstatq_11_OUT.pkl")
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
    infile = datadir("xstatq_12_IN.pkl")
    expfile = datadir("xstatq_12_OUT.pkl")
    dummy_test(infile, expfile)


def test_13():
    # similar to test_10 + mask added
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
    infile = datadir("statq_13_IN.pkl")
    expfile = datadir("statq_13_OUT.pkl")
    dummy_test(infile, expfile)


def test_14():
    # thickness_n10 data, slm and t_test run prior to fdr
    infile = datadir("xstatq_14_IN.pkl")
    expfile = datadir("xstatq_14_OUT.pkl")
    dummy_test(infile, expfile)
