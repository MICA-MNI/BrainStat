import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats._t_test import t_test
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    slm = SLM(FixedEffect(1), FixedEffect(1))
    for key in idic.keys():
        setattr(slm, key, idic[key])

    # run _t_test
    t_test(slm)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    testout = []
    for key in expdic.keys():
        comp = np.allclose(getattr(slm, key), expdic[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    # ['X'] : np array, shape (3, 1), float64
    # ['df']: np array, shape (1,), float64
    # ['coef'] : np array, shape (1, 3), float64
    # ['SSE'] : np array, shape (1, 3), float64
    # ['contrast'] : np array, shape (1, 1), float64
    infile = datadir("xstatt_01_IN.pkl")
    expfile = datadir("xstatt_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    # ['X'] : np array, shape (6, 6), float64
    # ['df']: np array, shape (1, 1), int64
    # ['coef'] : np array, shape (6, 6), float64
    # ['SSE'] : np array, shape (1, 6), float64
    # ['contrast']: np array, shape (1, 6), float64
    infile = datadir("xstatt_02_IN.pkl")
    expfile = datadir("xstatt_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    # ['X'] : np array, shape (3, 2), int64
    # ['V'] : np array, shape (3, 3, 2), float64
    # ['df'] : np array, shape (1,), float64
    # ['coef'] : np array, shape (2, 1), int64
    # ['SSE'] : np array, shape (1, 1), int64
    # ['r'] : np array, shape (1, 1), int64
    # ['dr'] np array, shape (1, 1), int64
    # ['contrast']: np array, shape (1, 1), int64
    # ['c']: np array, shape (1, 2), float64
    # ['k']: int
    # ['dfs']: np array, shape (1, 1), float64
    # ['ef']: np array, shape (1, 1), float64
    # ['sd']: np array, shape (1, 1), float64
    # ['t']: np array, shape (1, 1), float64
    infile = datadir("xstatt_03_IN.pkl")
    expfile = datadir("xstatt_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    # ['X'] : np array, shape (3, 2), float64
    # ['V'] : np array, shape (3, 3), int64
    # ['df'] : np array, shape (1,), int64
    # ['coef'] : np array, shape (2, 1), float64
    # ['SSE'] : np array, shape (1,), int64
    # ['dr'] : np array, shape (1,), int64
    # ['contrast']: np array, shape (1, 1), int64
    infile = datadir("xstatt_04_IN.pkl")
    expfile = datadir("xstatt_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    # ['X'] : np array, shape (10, 2), uint8
    # ['df'] : uint8
    # ['coef'] : np array, shape (2, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['contrast']: np array, shape (10, 1), uint8
    infile = datadir("xstatt_05_IN.pkl")
    expfile = datadir("xstatt_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    # ['X'] : np array, shape (10, 2), uint8
    # ['df'] : uint8
    # ['coef'] : np array, shape (2, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['contrast']: np array, shape (10, 1), int16
    infile = datadir("xstatt_06_IN.pkl")
    expfile = datadir("xstatt_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    # ['df'] : int64
    # ['X'] : np array, shape (10, 2), float64
    # ['coef'] : np array, shape (2, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['contrast']: np array, shape (10, 1), float64
    infile = datadir("xstatt_07_IN.pkl")
    expfile = datadir("xstatt_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    # ['X'] : np array, shape (20, 9), uint16
    # ['df'] : uint8
    # ['coef'] : np array, shape (9, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['contrast'] : np array, shape (20, 1), int64
    infile = datadir("xstatt_08_IN.pkl")
    expfile = datadir("xstatt_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    # ['df'] : int64
    # ['X'] : np array, shape (20, 9), uint16
    # ['coef'] : np array, shape (9, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['contrast'] : np array, shape (20, 1), int64
    infile = datadir("xstatt_09_IN.pkl")
    expfile = datadir("xstatt_09_OUT.pkl")
    dummy_test(infile, expfile)


def test_10():
    # ['df'] : int64
    # ['X'] : np array, shape (20, 9), uint16
    # ['coef'] : np array, shape (9, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['contrast'] : np array, shape (20, 1), int64
    infile = datadir("xstatt_10_IN.pkl")
    expfile = datadir("xstatt_10_OUT.pkl")
    dummy_test(infile, expfile)


def test_11():
    # ['df'] : int64
    # ['X'] : np array, shape (20, 9), uint16
    # ['coef'] : np array, shape (9, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['contrast'] : np array, shape (20, 1), int64
    infile = datadir("xstatt_11_IN.pkl")
    expfile = datadir("xstatt_11_OUT.pkl")
    dummy_test(infile, expfile)


def test_12():
    # ['df'] : int64
    # ['X'] : np array, shape (20, 9), uint16
    # ['coef'] : np array, shape (9, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['contrast'] : np array, shape (20, 1), int64
    infile = datadir("xstatt_12_IN.pkl")
    expfile = datadir("xstatt_12_OUT.pkl")
    dummy_test(infile, expfile)
