import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats.models import f_test


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    slm1 = {}
    slm1["X"] = idic["slm1X"]
    slm1["df"] = idic["slm1df"]
    slm1["SSE"] = idic["slm1SSE"]
    slm1["coef"] = idic["slm1coef"]

    slm2 = {}
    slm2["X"] = idic["slm2X"]
    slm2["df"] = idic["slm2df"]
    slm2["SSE"] = idic["slm2SSE"]
    slm2["coef"] = idic["slm2coef"]

    # check other potential keys in input
    if "slm1tri" in idic.keys():
        slm1["tri"] = idic["slm1tri"]
        slm2["tri"] = idic["slm2tri"]

    if "slm1resl" in idic.keys():
        slm1["resl"] = idic["slm1resl"]
        slm2["resl"] = idic["slm2resl"]

    if "slm1c" in idic.keys():
        slm1["c"] = idic["slm1c"]
        slm2["c"] = idic["slm2c"]

    if "slm1k" in idic.keys():
        slm1["k"] = idic["slm1k"]
        slm2["k"] = idic["slm2k"]

    if "slm1ef" in idic.keys():
        slm1["ef"] = idic["slm1ef"]
        slm2["ef"] = idic["slm2ef"]

    if "slm1sd" in idic.keys():
        slm1["sd"] = idic["slm1sd"]
        slm2["sd"] = idic["slm2sd"]

    if "slm1t" in idic.keys():
        slm1["t"] = idic["slm1t"]
        slm2["t"] = idic["slm2t"]

    # run f test
    outdic = f_test(slm1, slm2)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in outdic.keys():
        comp = np.allclose(outdic[key], expdic[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    assert all(flag == True for (flag) in testout)


# test data *pkl consists of slm1* and slm2* keys
# slm1* variables will be assigned to slm1 dictionary, and slm2* to the slm2 dict.


def test_01():
    # small sized slm1 and slm2 keys ['X'], ['df'], ['SSE'], ['coef']
    # slm1X : np array, shape (5, 6), int64
    # slm1df  : int
    # slm1SSE : np array, shape (3, 1), int64
    # slm1coef : np array, shape (6, 1), int64
    # slm2X' : np array, shape (5, 6), int64
    # slm2df' : int,
    # slm2SSE' : np array, shape (3, 1), int64
    # slm2coef' : np array, shape (6, 1), int64
    infile = datadir("statf_01_IN.pkl")
    expfile = datadir("statf_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    # middle sized slm1 and slm2 keys ['X'], ['df'], ['SSE'], ['coef']
    # slm1X : np array, shape (84, 77), int64
    # slm1df : int
    # slm1SSE : np array, shape (1128, 42), int64
    # slm1coef : np array, shape (77, 42), int64
    # slm2X : np array, shape (84, 77), int64
    # slm2df : int
    # slm2SSE : np array, shape (1128, 42), int64
    # slm2coef : np array, shape (77, 42), int64
    infile = datadir("statf_02_IN.pkl")
    expfile = datadir("statf_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    # middle sized slm1 and slm2 keys ['X'], ['df'], ['SSE'] + and ['SSE'] has 2k rows
    # slm1X np array, shape (91, 58), float64
    # slm1df : int
    # slm1SSE : np array, shape (2278, 75), float64
    # slm1coef : np array, shape (58, 75), float64
    # slm2X : np array, shape (91, 58), float64
    # slm2df : int
    # slm2SSE : np array, shape (2278, 75), float64
    # slm2coef : np array, shape (58, 75) float64
    infile = datadir("statf_03_IN.pkl")
    expfile = datadir("statf_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    # small sized input slm1 and slm2 keys ['X'], ['df'], ['SSE'], ['coef'] is 3D
    # slm1X : np array, shape (19, 27), int64
    # slm1df : int
    # slm1SSE : np array, shape (6, 87), int64
    # slm1coef : np array, shape (27, 87, 3), float64
    # slm2X : np array, shape (19, 27), int64
    # slm2df : int
    # slm2SSE : np array, shape (6, 87), int64
    # slm2coef : np array, shape (27, 87, 3), float64
    infile = datadir("statf_04_IN.pkl")
    expfile = datadir("statf_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    # similar to test_04, shapes of ['X'], ['SSE'] and ['coef'] changed
    # slm1X : np array, shape (13, 3), int64
    # slm1df : int
    # slm1SSE : np array, shape (3, 27), int64
    # slm1coef : np array, shape (3, 27, 2), float64
    # slm2X : np array, shape (13, 3),  int64
    # slm2df : int
    # slm2SSE : np array, shape (3, 27), int64
    # slm2coef np array, shape (3, 27, 2), float64
    infile = datadir("statf_05_IN.pkl")
    expfile = datadir("statf_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    # similar to test_04, shapes/values of ['X'], ['SSE'], ['df'] and ['coef'] changed
    # slm1X : np array, shape (13, 10), int64
    # slm1df : int
    # slm1SSE : np array, shape (3, 34), int64
    # slm1coef : np array, shape (10, 34, 2), int64
    # slm2X : np array, shape (13, 10), int64
    # slm2df : int
    # slm2SSE : np array, shape (3, 34), int64
    # slm2coef : np array, shape (10, 34, 2), int64
    infile = datadir("statf_06_IN.pkl")
    expfile = datadir("statf_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    # similar to test_04, shapes/values of ['X'], ['SSE'], ['df'] and ['coef'] changed
    # slm1X : np array, shape (12, 4), float64
    # slm1df : int
    # slm1SSE : np array, shape (6, 42), float64
    # slm1coef : np array, shape (4, 42, 3), float64
    # slm2X : np array, shape (12, 4), float64
    # slm2df : int
    # slm2SSE np array, shape (6, 42), float64
    # slm2coef np array, shape (4, 42, 3), float64
    infile = datadir("statf_07_IN.pkl")
    expfile = datadir("statf_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    # similar to test_04, shapes/values of ['X'], ['SSE'], ['df'] and ['coef'] changed
    # slm1X : np array, shape (32, 91), float64
    # slm1df : int
    # slm1SSE : np array, shape (3, 78), float64
    # slm1coef : np array, shape (91, 78, 2), float64
    # slm2X : np array, shape (32, 91), float64
    # slm2df : int
    # slm2SSE np array, shape (3, 78), float64
    # slm2coef np array, shape (91, 78, 2), float64
    infile = datadir("statf_08_IN.pkl")
    expfile = datadir("statf_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    # similar to test_04, shapes/values of ['X'], ['SSE'], ['df'] and ['coef'] changed
    # slm1X : np array, shape (88, 49), float64
    # slm1df : int
    # slm1SSE : np array, shape (1, 56), float64
    # slm1coef : np array, shape (49, 56, 1), float64
    # slm2X : np array, shape (88, 49), float64
    # slm2df : int
    # slm2SSE : np array, shape (1, 56), float64
    # slm2coef : np array, shape (49, 56, 1), float64
    infile = datadir("statf_09_IN.pkl")
    expfile = datadir("statf_09_OUT.pkl")
    dummy_test(infile, expfile)


def test_10():
    # real data set with more keys then ['X'], ['SSE'], ['df'] and ['coef'] given
    # slm1X : np array, shape (10, 2), uint8
    # slm1df : int
    # slm1SSE : np array, shape (1, 20484), float64
    # slm1coef : np array, shape (2, 20484), float64
    # slm1tri : np array, shape (40960, 3), int32
    # slm1resl : np array, shape (61440, 1), float64
    # slm1c : np array, shape (1, 2), float64
    # slm1k : int
    # slm1ef : np array, shape (1, 20484), float64
    # slm1sd : np array, shape (1, 20484), float64
    # slm1t : np array, shape (1, 20484), float64
    # slm2X : np array, shape (10, 2), uint8
    # slm2df : int
    # slm2SSE : np array, shape (1, 20484), float64
    # slm2coef : np array, shape (2, 20484), float64
    # slm2tri : np array, shape (40960, 3), int32
    # slm2resl : np array, shape (61440, 1), float64
    # slm2c : np array, shape (1, 2), float64
    # slm2k : int
    # slm2ef : np array, shape (1, 20484), float64
    # slm2sd : np array, shape (1, 20484), float64
    # slm2t : np array, shape (1, 20484), float64
    infile = datadir("statf_10_IN.pkl")
    expfile = datadir("statf_10_OUT.pkl")
    dummy_test(infile, expfile)


def test_11():
    # test_10 + slm2['resl'] and slm2[X] shuffled, slm2['df'] changed
    # slm1X : np array, shape (10, 2), uint8
    # slm1df : int
    # slm1SSE : np array, shape (1, 20484), float64
    # slm1coef : np array, shape (2, 20484), float64
    # slm1tri : np array, shape (40960, 3), int32
    # slm1resl : np array, shape (61440, 1), float64
    # slm1c : np array, shape (1, 2), float64
    # slm1k : int
    # slm1ef : np array, shape (1, 20484), float64
    # slm1sd : np array, shape (1, 20484), float64
    # slm1t : np array, shape (1, 20484), float64
    # slm2X : np array, shape (10, 2), uint8
    # slm2df : int
    # slm2SSE : np array, shape (1, 20484), float64
    # slm2coef : np array, shape (2, 20484), float64
    # slm2tri : np array, shape (40960, 3), int32
    # slm2resl : np array, shape (61440, 1), float64
    # slm2c : np array, shape (1, 2), float64
    # slm2k : int
    # slm2ef : np array, shape (1, 20484), float64
    # slm2sd : np array, shape (1, 20484), float64
    # slm2t : np array, shape (1, 20484), float64
    infile = datadir("statf_11_IN.pkl")
    expfile = datadir("statf_11_OUT.pkl")
    dummy_test(infile, expfile)


def test_12():
    # test_10 + shapes/values of ['X'], ['SSE'], ['df'] and ['coef'] changed
    # slm1X : np array, shape (20, 9), uint16
    # slm1df : int
    # slm1SSE : np array, shape (1, 20484), float64
    # slm1coef : np array, shape (9, 20484), float64
    # slm1tri : np array, shape (40960, 3), int32
    # slm1resl : np array, shape (61440, 1), float64
    # slm1c : np array, shape (1, 9), float64
    # slm1k : int
    # slm1ef : np array, shape (1, 20484), float64
    # slm1sd : np array, shape (1, 20484), float64
    # slm1t : np array, shape (1, 20484), float64
    # slm2X : np array, shape (20, 9), uint16
    # slm2df : int
    # slm2SSE : np array, shape (1, 20484), float64
    # slm2coef : np array, shape (9, 20484), float64
    # slm2tri : np array, shape (40960, 3), int32
    # slm2resl : np array, shape (61440, 1), float64
    # slm2c : np array, shape (1, 9), float64
    # slm2k : int
    # slm2ef : np array, shape (1, 20484), float64
    # slm2sd : np array, shape (1, 20484), float64
    # slm2t : np array, shape (1, 20484), float64
    infile = datadir("statf_12_IN.pkl")
    expfile = datadir("statf_12_OUT.pkl")
    dummy_test(infile, expfile)
