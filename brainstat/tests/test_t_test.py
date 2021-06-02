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
    infile = datadir("xstatt_01_IN.pkl")
    expfile = datadir("xstatt_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    infile = datadir("xstatt_02_IN.pkl")
    expfile = datadir("xstatt_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    infile = datadir("xstatt_03_IN.pkl")
    expfile = datadir("xstatt_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    infile = datadir("xstatt_04_IN.pkl")
    expfile = datadir("xstatt_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    infile = datadir("xstatt_05_IN.pkl")
    expfile = datadir("xstatt_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    infile = datadir("xstatt_06_IN.pkl")
    expfile = datadir("xstatt_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    infile = datadir("xstatt_07_IN.pkl")
    expfile = datadir("xstatt_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    infile = datadir("xstatt_08_IN.pkl")
    expfile = datadir("xstatt_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    infile = datadir("xstatt_09_IN.pkl")
    expfile = datadir("xstatt_09_OUT.pkl")
    dummy_test(infile, expfile)


def test_10():
    infile = datadir("xstatt_10_IN.pkl")
    expfile = datadir("xstatt_10_OUT.pkl")
    dummy_test(infile, expfile)


def test_11():
    infile = datadir("xstatt_11_IN.pkl")
    expfile = datadir("xstatt_11_OUT.pkl")
    dummy_test(infile, expfile)


def test_12():
    infile = datadir("xstatt_12_IN.pkl")
    expfile = datadir("xstatt_12_OUT.pkl")
    dummy_test(infile, expfile)


def test_13():
    infile = datadir("xstatt_13_IN.pkl")
    expfile = datadir("xstatt_13_OUT.pkl")
    dummy_test(infile, expfile)
