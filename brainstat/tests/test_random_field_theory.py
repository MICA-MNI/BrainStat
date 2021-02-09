import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats._multiple_comparisons import random_field_theory
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import Term

def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    slm = SLM(Term(1), Term(1))
    for key in idic.keys():
        if key is 'clusthresh':
            slm.cluster_threshold = idic[key]
        else:
            setattr(slm, key, idic[key])

    PY_pval, PY_peak, PY_clus, PY_clusid = random_field_theory(slm)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    O_pval = expdic["pval"]
    O_peak = expdic["peak"]
    O_clus = expdic["clus"]
    O_clusid = expdic["clusid"]

    testout = []

    for key in PY_pval.keys():
        comp = np.allclose(PY_pval[key], O_pval[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    if isinstance(PY_peak, dict):
        for key in PY_peak.keys():
            comp = np.allclose(PY_peak[key], O_peak[key], rtol=1e-05, equal_nan=True)
    else:
        comp = np.allclose(PY_peak, O_peak, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    if isinstance(PY_peak, dict):
        for key in PY_clus.keys():
            comp = np.allclose(PY_clus[key], O_clus[key], rtol=1e-05, equal_nan=True)
    else:
        comp = np.allclose(PY_clus, O_clus, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    testout.append(np.allclose(PY_clusid, O_clusid, rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout)


def test_01():
    # slm with only one vertex with t-value ['t'], huge sized ['tri'] and ['resl']
    # ['t'] : np array, shape (1, 1), float64
    # ['df'] : np array, shape (1, 1), uint16
    # ['k'] : np array, shape (1, 1), uint8
    # ['tri'] : np array, shape (129960, 3), int32
    # ['resl'] : np array, shape (194940, 1), float64
    # ['dfs'] : np array, shape (1, 1), int64
    infile = datadir("statp_01_IN.pkl")
    expfile = datadir("statp_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    # slm with intermediate sized ['t'], huge sized ['tri'] and ['resl']
    # ['t'] : np array, shape (2483, 1), float64
    # ['df'] : np array, shape (1, 1), uint16
    # ['k'] : np array, shape (1, 1), uint8
    # ['resl'] : np array, shape (194940, 1), float64
    # ['tri'] : np array, shape (129960, 3), int32
    # ['dfs'] : np array, shape (1, 1), int64
    infile = datadir("statp_02_IN.pkl")
    expfile = datadir("statp_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    # slm['k'] data type changed to an int
    # ['t'] : np array, shape (5969, 1), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : int
    # ['resl'] : np array, shape (3, 1), float64
    # ['tri'] : np array, shape (129960, 3), int32
    # ['dfs'] : np array, shape (1, 1), int64
    infile = datadir("statp_03_IN.pkl")
    expfile = datadir("statp_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    # only mandatory keys of slm are given, slm['t'] is 64k
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] :  np array, shape (1, 1), uint16
    # ['k'] :  np array, shape (1, 1), uint8
    # ['resl'] :  np array, shape (194940, 1), float64
    # ['tri'] :  np array, shape (129960, 3), int32
    infile = datadir("statp_04_IN.pkl")
    expfile = datadir("statp_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    # only mandatory slm keys given, change dtype of slm['df'] and slm['k']
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] :  int
    # ['resl'] : np array, shape (194940, 1), float64
    # ['tri'] : np array, shape (129960, 3), int32
    infile = datadir("statp_05_IN.pkl")
    expfile = datadir("statp_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    # only mandatory slm keys given, change dtype of slm['df'] and slm['k']
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] : np array, shape (1, 1), uint16
    # ['k'] : np array, shape (1, 1), uint8
    # ['resl'] : np array, shape (194940, 1), float64
    # ['tri'] : np array, shape (129960, 3), int32
    infile = datadir("statp_06_IN.pkl")
    expfile = datadir("statp_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    # only mandatory keys of slm are given, change dtype of slm['df'] and slm['k']
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] : np array, shape (1, 1), int64
    # ['k'] : np array, shape (1, 1), uint8
    # ['resl'] : np array, shape (194940, 1), float64
    # ['tri'] : np array, shape (129960, 3), int32
    infile = datadir("statp_07_IN.pkl")
    expfile = datadir("statp_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    # values in slm['t'] (of test_07) were shuffled
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] : np array, shape (1, 1), uint16
    # ['k'] :  np array, shape (1, 1), uint8
    # ['resl'] : np array, shape (194940, 1), float64
    # ['tri'] : np array, shape (129960, 3), int32
    infile = datadir("statp_08_IN.pkl")
    expfile = datadir("statp_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    # values in slm['t'] array (of test_07) were shuffled
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] : np array, shape (1, 1), uint16
    # ['k'] : np array, shape (1, 1), uint8
    # ['resl'] : np array, shape (194940, 1), float64
    # ['tri'] : np array, shape (129960, 3), int32
    infile = datadir("statp_09_IN.pkl")
    expfile = datadir("statp_09_OUT.pkl")
    dummy_test(infile, expfile)


def test_10():
    # only mandatory slm keys given + optional input ['mask'] given
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] : np array, shape (1, 1), uint16
    # ['k'] : np array, shape (1, 1), uint8
    # ['resl'] : np array, shape (194940, 1), float64
    # ['tri'] : np array, shape (129960, 3), int32
    # ['mask'] : np array, shape (64984,), bool
    infile = datadir("statp_10_IN.pkl")
    expfile = datadir("statp_10_OUT.pkl")
    dummy_test(infile, expfile)


def test_11():
    # test_10 + optional input ['mask'] and ['clusthresh'] given
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] : np array, shape (1, 1), uint16
    # ['k'] : np array, shape (1, 1), uint8
    # ['resl'] : np array, shape (194940, 1), float64
    # ['tri'] : np array, shape (129960, 3), int32
    # ['mask'] : mask np array, shape (64984,), bool
    # ['clusthresh'] : <class 'float'>
    infile = datadir("statp_11_IN.pkl")
    expfile = datadir("statp_11_OUT.pkl")
    dummy_test(infile, expfile)


def test_13():
    # only mandatory slm keys given, slm['df'] is changed to a 1D array
    # ['t'] : np array, shape (1, 64984), float64
    # ['df'] : np array, shape (1,), int64
    # ['k'] : int
    # ['resl'] np array, shape (194940, 1), float64
    # ['tri']  np array, shape (129960, 3), int32
    infile = datadir("statp_13_IN.pkl")
    expfile = datadir("statp_13_OUT.pkl")
    dummy_test(infile, expfile)


def test_14():
    # test_04 + optional slm['dfs'] (huge sized) given
    # ['t'] :  np array, shape (1, 64984), float64
    # ['df'] : np array, shape (1, 1), uint16
    # ['k'] : np array, shape (1, 1), uint8
    # ['resl'] : np array, shape (194940, 1), float64
    # ['tri'] : np array, shape (129960, 3), int32
    # ['dfs'] : np array, shape (1, 64984), int64
    infile = datadir("statp_14_IN.pkl")
    expfile = datadir("statp_14_OUT.pkl")
    dummy_test(infile, expfile)


def test_15():
    # only mandatory slm keys are given, ['t'] is 32k, half-sized of test_04
    # ['t'] : np array, shape (1, 32492), float64
    # ['df'] : np array, shape (1, 1), uint8
    # ['k'] : np array, shape (1, 1), uint8
    # ['resl'] : np array, shape (97470, 3), float64
    # ['tri'] : np array, shape (64980, 3), int32
    infile = datadir("statp_15_IN.pkl")
    expfile = datadir("statp_15_OUT.pkl")
    dummy_test(infile, expfile)


def test_16():
    # additional non-sense keys (meaningless for _random_field_theory) are added to slm
    # ['df'] : int64
    # ['X'] : np array, shape (10, 2), float64
    # ['coef'] :  np array, shape (2, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['c'] : np array, shape (1, 2), float64
    # ['k'] : int
    # ['ef'] : np array, shape (1, 20484), float64
    # ['sd'] : np array, shape (1, 20484), float64
    # ['t'] :  np array, shape (1, 20484), float64
    infile = datadir("statp_16_IN.pkl")
    expfile = datadir("statp_16_OUT.pkl")
    dummy_test(infile, expfile)


def test_17():
    # test_16 + dtype change of non-sense slm keys + optional ['mask'] input given
    # ['X'] : np array, shape (10, 2), uint8
    # ['df'] : uint8
    # ['coef'] : np array, shape (2, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] :  np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['c'] :  np array, shape (1, 2), float64
    # ['k'] :  int
    # ['ef'] : np array, shape (1, 20484), float64
    # ['sd'] : np array, shape (1, 20484), float64
    # ['t'] :  np array, shape (1, 20484), float64
    # ['mask'] :  np array, shape (20484,), bool
    infile = datadir("statp_17_IN.pkl")
    expfile = datadir("statp_17_OUT.pkl")
    dummy_test(infile, expfile)


def test_18():
    # test_16 + dtype/shape change of the non-sense slm keys
    # ['df'] :  int64
    # ['X'] np array, shape (20, 9), uint16
    # ['coef'] : np array, shape (9, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['c'] : np array, shape (1, 9), float64
    # ['k'] : int
    # ['ef'] : np array, shape (1, 20484), float64
    # ['sd'] : np array, shape (1, 20484), float64
    # ['t'] :  np array, shape (1, 20484), float64
    infile = datadir("statp_18_IN.pkl")
    expfile = datadir("statp_18_OUT.pkl")
    dummy_test(infile, expfile)


def test_19():
    # test_16 + dtype/shape change of non-sense slm keys  + optional ['mask'] input
    # ['X'] : np array, shape (20, 9), uint16
    # ['df'] :  uint8
    # ['coef'] : np array, shape (9, 20484), float64
    # ['SSE'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['resl'] : np array, shape (61440, 1), float64
    # ['c'] : np array, shape (1, 9), float64
    # ['k'] : int
    # ['ef'] : np array, shape (1, 20484), float64
    # ['sd'] : np array, shape (1, 20484), float64
    # ['t'] :  np array, shape (1, 20484), float64
    # ['mask'] : np array, shape (20484,), bool
    infile = datadir("statp_19_IN.pkl")
    expfile = datadir("statp_19_OUT.pkl")
    dummy_test(infile, expfile)
