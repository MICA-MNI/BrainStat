import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats._multiple_comparisons import peak_clus
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import Term


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    slm = SLM(Term(1), Term(1))
    slm.t = idic["t"]
    slm.tri = idic["tri"]
    slm.mask = idic["mask"]
    thresh = idic["thresh"]
    reselspvert = None
    edg = None

    if "reselspvert" in idic.keys():
        reselspvert = idic["reselspvert"]

    if "edg" in idic.keys():
        edg = idic["edg"]

    if "k" in idic.keys():
        slm.k = idic["k"]

    if "df" in idic.keys():
        slm.df = idic["df"]

    # call python function
    P_peak, P_clus, P_clusid = peak_clus(slm, thresh, reselspvert, edg)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    O_peak = expdic["peak"]
    O_clus = expdic["clus"]
    O_clusid = expdic["clusid"]

    testout = []

    if isinstance(P_peak, dict):
        for key in P_peak.keys():
            comp = np.allclose(P_peak[key], O_peak[key], rtol=1e-05, equal_nan=True)
            testout.append(comp)
    else:
        comp = np.allclose(P_peak, O_peak, rtol=1e-05, equal_nan=True)

    if isinstance(P_clus, dict):
        for key in P_clus.keys():
            comp = np.allclose(P_clus[key], O_clus[key], rtol=1e-05, equal_nan=True)
    else:
        comp = np.allclose(P_clus, O_clus, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    testout.append(np.allclose(P_clusid, O_clusid, rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout)


def test_01():
    # real-data testing; data to be assigned to slm['t'], slm['tri'], mask and thresh
    # ['t'] : np array, shape (1, 64984), float64
    # ['tri] : np array, shape (129960, 3), int32
    # ['mask'] : np array, shape (64984,), float64
    # ['thresh'] : float
    infile = datadir("statpeakc_01_IN.pkl")
    expfile = datadir("statpeakc_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    # non-sensical input; test data with more vertices than exists
    # ['t'] : np array, shape (1, 1000), float64
    # ['tri] : np array, shape (100, 3), int64
    # ['mask'] : np array, shape (1000,), float64
    # ['thresh'] : float
    infile = datadir("statpeakc_02_IN.pkl")
    expfile = datadir("statpeakc_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    # non-sensical input; test data with more vertices than exists
    # ['t'] : np array, shape (1, 598), float64
    # ['tri] : np array, shape (330, 3), int64
    # ['mask'] : np array, shape (598,), float64
    # ['thresh'] : float
    infile = datadir("statpeakc_03_IN.pkl")
    expfile = datadir("statpeakc_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    # non-sensical input; test data with more vertices than exists
    # ['t'] :  np array, shape (1, 8961), float64
    # ['tri] : np array, shape (4171, 3), int64
    # ['mask'] : np array, shape (8961,), float64
    # ['thresh'] : float
    infile = datadir("statpeakc_04_IN.pkl")
    expfile = datadir("statpeakc_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    # similar to test_01 + optional input c
    # ['t'] : np array, shape (1, 64984), float64
    # ['tri] : np array, shape (129960, 3), int32
    # ['mask'] : np array, shape (64984,), float64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (64984,), float64
    infile = datadir("statpeakc_05_IN.pkl")
    expfile = datadir("statpeakc_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    # artifical data for slm['t'], slm['tri'], ['mask'], ['thresh'], ['reselspvert']
    # ['t'] : np array, shape (1, 5926), float64
    # ['tri] : np array, shape (8467, 3), int64
    # ['mask'] : np array, shape (5926,), float64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (5926,), float64
    infile = datadir("statpeakc_06_IN.pkl")
    expfile = datadir("statpeakc_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    # similar to test_06 + optional input slm['k'] and slm['df']
    # ['t'] : np array, shape (1, 4593), float64
    # ['tri] :  np array, shape (8181, 3), int64
    # ['mask'] : np array, shape (4593,), float64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (4593,), float64
    # ['k'] : int
    # ['df'] : int
    infile = datadir("statpeakc_07_IN.pkl")
    expfile = datadir("statpeakc_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    # similar to test_07, shape/values of input params changed
    # ['t'] : np array, shape (2, 4496), float64
    # ['tri] :  np array, shape (7793, 3), int64
    # ['mask'] : np array, shape (4496,), float64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (4496,), float64
    # ['k'] : int
    # ['df'] : int
    infile = datadir("statpeakc_08_IN.pkl")
    expfile = datadir("statpeakc_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    # similar to test_07, shape/values of input params changed
    # ['t'] : np array, shape (2, 4085), float64
    # ['tri] : np array, shape (4673, 3), int64
    # ['mask'] : np array, shape (4085,), float64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (4085,), float64
    # ['k'] : int
    # ['df'] : int
    infile = datadir("statpeakc_09_IN.pkl")
    expfile = datadir("statpeakc_09_OUT.pkl")
    dummy_test(infile, expfile)


def test_10():
    # similar to test_07, shape/values of input params changed
    # ['t'] : np array, shape (1, 8594), float64
    # ['tri] : np array, shape (9770, 3), int64
    # ['mask'] : np array, shape (8594,), float64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (8594,), float64
    # ['k'] : int
    # ['df'] : int
    infile = datadir("statpeakc_10_IN.pkl")
    expfile = datadir("statpeakc_10_OUT.pkl")
    dummy_test(infile, expfile)


def test_11():
    # similar to test_07, shape/values of input params changed
    # ['t'] :  np array, shape (2, 4225), float64
    # ['tri'] : np array, shape (8820, 3), int64
    # ['mask'] :  np array, shape (4225,), float64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (4225,), float64
    # ['k'] : int
    # ['df'] : int
    infile = datadir("statpeakc_11_IN.pkl")
    expfile = datadir("statpeakc_11_OUT.pkl")
    dummy_test(infile, expfile)


def test_12():
    # similar to test_07, shape/values of input params changed
    # ['t'] : np array, shape (3, 7534), float64
    # ['tri'] : np array, shape (3190, 3), int64
    # ['mask'] : np array, shape (7534,), float64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (7534,), float64
    # ['k'] : int
    # ['df'] : int
    infile = datadir("statpeakc_12_IN.pkl")
    expfile = datadir("statpeakc_12_OUT.pkl")
    dummy_test(infile, expfile)


def test_13():
    # similar to test_06, shape/values of input params changed
    # ['t'] : np array, shape (1, 9550), float64
    # ['tri'] : np array, shape (2891, 3), int64
    # ['mask'] : np array, shape (9550,), int64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (9550,), float64
    infile = datadir("statpeakc_13_IN.pkl")
    expfile = datadir("statpeakc_13_OUT.pkl")
    dummy_test(infile, expfile)


def test_14():
    # similar to test_07, shape/values of input params changed
    # ['t'] : np array, shape (2, 6550), float64
    # ['tri'] :  np array, shape (8049, 3), int64
    # ['mask'] : np array, shape (6550,), int64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (6550,), float64
    # ['k'] : int
    # ['df'] : int
    infile = datadir("statpeakc_14_IN.pkl")
    expfile = datadir("statpeakc_14_OUT.pkl")
    dummy_test(infile, expfile)


def test_15():
    # similar to test_06 + optional ['edg'] input
    # ['t'] : np array, shape (1, 1000), float64
    # ['tri'] : np array, shape (100, 3), int64
    # ['mask'] : np array, shape (1000,), float64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (1000,), float64
    # ['edg'] : np array, shape (300, 2), int64
    infile = datadir("statpeakc_15_IN.pkl")
    expfile = datadir("statpeakc_15_OUT.pkl")
    dummy_test(infile, expfile)


def test_16():
    # similar to test_07 + optional ['edg'] input
    # ['t'] : np array, shape (2, 9521), float64
    # ['tri'] :  np array, shape (6660, 3), int64
    # ['mask'] : np array, shape (9521,), int64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (9521,), float64
    # ['k'] : int
    # ['df'] : int
    # ['edg'] : np array, shape (19977, 2), int64
    infile = datadir("statpeakc_16_IN.pkl")
    expfile = datadir("statpeakc_16_OUT.pkl")
    dummy_test(infile, expfile)


def test_17():
    # similar to test_15, shape of ['edg'] changed
    # ['t'] : np array, shape (1, 1000), float64
    # ['tri'] :  np array, shape (100, 3), int64
    # ['mask'] : np array, shape (1000,), int64
    # ['thresh'] : float
    # ['reselspvert'] : np array, shape (1000,), float64
    # ['edg'] : np array, shape (1228, 2), int64
    infile = datadir("statpeakc_17_IN.pkl")
    expfile = datadir("statpeakc_17_OUT.pkl")
    dummy_test(infile, expfile)


def test_18():
    # non-sensical input (similar to test_02), ['thresh'] dtype changed
    # ['t'] : np array, shape (1, 1000), float64
    # ['tri'] : np array, shape (100, 3), int64
    # ['mask'] : np array, shape (1000,), float64
    # ['thresh'] : int
    infile = datadir("statpeakc_18_IN.pkl")
    expfile = datadir("statpeakc_18_OUT.pkl")
    dummy_test(infile, expfile)


def test_19():
    # similar to test_01, ['thresh'] dtype changed
    # ['t'] : np array, shape (1, 64984), float64
    # ['tri'] : np array, shape (129960, 3), int32
    # ['mask'] : np array, shape (64984,), float64
    # ['thresh'] : int
    infile = datadir("statpeakc_19_IN.pkl")
    expfile = datadir("statpeakc_19_OUT.pkl")
    dummy_test(infile, expfile)


def test_20():
    # similar to test_15, shape of ['edg'] changed
    # ['t'] : np array, shape (1, 1000), float64
    # ['tri'] : np array, shape (100, 3), int64
    # ['mask'] : np array, shape (1000,), int64
    # ['thresh'] : int
    # ['reselspvert'] :  np array, shape (1000,), float64
    # ['edg'] : np array, shape (1312, 2), int64
    infile = datadir("statpeakc_20_IN.pkl")
    expfile = datadir("statpeakc_20_OUT.pkl")
    dummy_test(infile, expfile)
