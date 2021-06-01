import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats._multiple_comparisons import peak_clus
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    slm = SLM(FixedEffect(1), FixedEffect(1))
    slm.t = idic["t"]
    slm.tri = idic["tri"]
    slm.mask = idic["mask"]
    slm.df = idic["df"]
    slm.k = idic["k"]
    slm.dfs = idic["dfs"]
    slm.resl = idic["resl"]

    thresh = idic["thresh"]
    reselspvert = idic["reselspvert"]
    edg = idic["edg"]

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
    infile = datadir("xstatpeakc_01_IN.pkl")
    expfile = datadir("xstatpeakc_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    infile = datadir("xstatpeakc_02_IN.pkl")
    expfile = datadir("xstatpeakc_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    infile = datadir("xstatpeakc_03_IN.pkl")
    expfile = datadir("xstatpeakc_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    infile = datadir("xstatpeakc_04_IN.pkl")
    expfile = datadir("xstatpeakc_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    infile = datadir("xstatpeakc_05_IN.pkl")
    expfile = datadir("xstatpeakc_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    infile = datadir("xstatpeakc_06_IN.pkl")
    expfile = datadir("xstatpeakc_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    infile = datadir("xstatpeakc_07_IN.pkl")
    expfile = datadir("xstatpeakc_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    infile = datadir("xstatpeakc_08_IN.pkl")
    expfile = datadir("xstatpeakc_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    infile = datadir("xstatpeakc_09_IN.pkl")
    expfile = datadir("xstatpeakc_09_OUT.pkl")
    dummy_test(infile, expfile)


def test_10():
    infile = datadir("xstatpeakc_10_IN.pkl")
    expfile = datadir("xstatpeakc_10_OUT.pkl")
    dummy_test(infile, expfile)


def test_11():
    infile = datadir("xstatpeakc_11_IN.pkl")
    expfile = datadir("xstatpeakc_11_OUT.pkl")
    dummy_test(infile, expfile)


def test_12():
    infile = datadir("xstatpeakc_12_IN.pkl")
    expfile = datadir("xstatpeakc_12_OUT.pkl")
    dummy_test(infile, expfile)


def test_13():
    infile = datadir("xstatpeakc_13_IN.pkl")
    expfile = datadir("xstatpeakc_13_OUT.pkl")
    dummy_test(infile, expfile)


def test_14():
    infile = datadir("xstatpeakc_14_IN.pkl")
    expfile = datadir("xstatpeakc_14_OUT.pkl")
    dummy_test(infile, expfile)


def test_15():
    infile = datadir("xstatpeakc_15_IN.pkl")
    expfile = datadir("xstatpeakc_15_OUT.pkl")
    dummy_test(infile, expfile)


def test_16():
    infile = datadir("xstatpeakc_16_IN.pkl")
    expfile = datadir("xstatpeakc_16_OUT.pkl")
    dummy_test(infile, expfile)


def test_17():
    infile = datadir("xstatpeakc_17_IN.pkl")
    expfile = datadir("xstatpeakc_17_OUT.pkl")
    dummy_test(infile, expfile)


def test_18():
    infile = datadir("xstatpeakc_18_IN.pkl")
    expfile = datadir("xstatpeakc_18_OUT.pkl")
    dummy_test(infile, expfile)


def test_19():
    infile = datadir("xstatpeakc_19_IN.pkl")
    expfile = datadir("xstatpeakc_19_OUT.pkl")
    dummy_test(infile, expfile)


def test_20():
    infile = datadir("xstatpeakc_20_IN.pkl")
    expfile = datadir("xstatpeakc_20_OUT.pkl")
    dummy_test(infile, expfile)


def test_21():
    infile = datadir("xstatpeakc_21_IN.pkl")
    expfile = datadir("xstatpeakc_21_OUT.pkl")
    dummy_test(infile, expfile)


def test_22():
    infile = datadir("xstatpeakc_22_IN.pkl")
    expfile = datadir("xstatpeakc_22_OUT.pkl")
    dummy_test(infile, expfile)


def test_23():
    infile = datadir("xstatpeakc_23_IN.pkl")
    expfile = datadir("xstatpeakc_23_OUT.pkl")
    dummy_test(infile, expfile)


def test_24():
    infile = datadir("xstatpeakc_24_IN.pkl")
    expfile = datadir("xstatpeakc_24_OUT.pkl")
    dummy_test(infile, expfile)
