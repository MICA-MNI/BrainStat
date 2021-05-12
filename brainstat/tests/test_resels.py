import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats._multiple_comparisons import compute_resels
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

    resels_py, reselspvert_py, edg_py = compute_resels(slm)

    out = {}
    out["resels"] = resels_py
    out["reselspvert"] = reselspvert_py
    out["edg"] = edg_py

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in out.keys():
        if out[key] is not None and expdic[key] is not None:
            comp = np.allclose(out[key], expdic[key], rtol=1e-05, equal_nan=True)
            testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    infile = datadir("xstatresl_01_IN.pkl")
    expfile = datadir("xstatresl_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    infile = datadir("xstatresl_02_IN.pkl")
    expfile = datadir("xstatresl_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    infile = datadir("xstatresl_03_IN.pkl")
    expfile = datadir("xstatresl_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    infile = datadir("xstatresl_04_IN.pkl")
    expfile = datadir("xstatresl_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    infile = datadir("xstatresl_05_IN.pkl")
    expfile = datadir("xstatresl_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    # ['lat'] :np array, shape (10, 10, 10), bool
    # ['mask'] :np array, shape (457,), bool
    infile = datadir("statresl_06_IN.pkl")
    expfile = datadir("statresl_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    # ['lat'] :np array, shape (10, 10, 10), bool
    # ['resl'] :np array, shape (1359, 1), float64
    infile = datadir("xstatresl_07_IN.pkl")
    expfile = datadir("xstatresl_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    # ['lat'] :np array, shape (10, 10, 10), bool
    # ['resl'] :np array, shape (1251, 1), float64
    # ['mask'] :np array, shape (499,), bool
    infile = datadir("statresl_08_IN.pkl")
    expfile = datadir("statresl_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    # ['lat'] :np array, shape (10, 10, 10), bool
    # ['resl'] :np array, shape (1198, 1), float64
    # ['mask'] :np array, shape (478,), bool
    infile = datadir("statresl_09_IN.pkl")
    expfile = datadir("statresl_09_OUT.pkl")
    dummy_test(infile, expfile)

#
#def test_10():
#    # ['tri'] :np array, shape (129960, 3), int32
#    # ['resl'] :np array, shape (194940, 1), float64
#    infile = datadir("statresl_10_IN.pkl")
#    expfile = datadir("statresl_10_OUT.pkl")
#    dummy_test(infile, expfile)
#
#
#def test_11():
#    # ['tri'] :np array, shape (129960, 3), int32
#    # ['resl'] :np array, shape (194940, 1), float64
#    # ['mask'] :np array, shape (64984,), bool
#    infile = datadir("statresl_11_IN.pkl")
#    expfile = datadir("statresl_11_OUT.pkl")
#    dummy_test(infile, expfile)
#
#
#def test_12():
#    # ['tri'] :np array, shape (129960, 3), int32
#    # ['resl'] :np array, shape (194940, 1), float64
#    infile = datadir("statresl_12_IN.pkl")
#    expfile = datadir("statresl_12_OUT.pkl")
#    dummy_test(infile, expfile)
#
#
#def test_13():
#    # ['tri'] :np array, shape (129960, 3), int32
#    # ['resl'] :np array, shape (194940, 1), float64
#    # ['mask'] :np array, shape (64984,), bool
#    infile = datadir("statresl_13_IN.pkl")
#    expfile = datadir("statresl_13_OUT.pkl")
#    dummy_test(infile, expfile)
