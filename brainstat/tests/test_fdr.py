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
    infile = datadir("xstatq_01_IN.pkl")
    expfile = datadir("xstatq_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    infile = datadir("xstatq_02_IN.pkl")
    expfile = datadir("xstatq_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    infile = datadir("xstatq_03_IN.pkl")
    expfile = datadir("xstatq_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    infile = datadir("xstatq_04_IN.pkl")
    expfile = datadir("xstatq_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    infile = datadir("xstatq_05_IN.pkl")
    expfile = datadir("xstatq_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    infile = datadir("xstatq_06_IN.pkl")
    expfile = datadir("xstatq_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    infile = datadir("xstatq_07_IN.pkl")
    expfile = datadir("xstatq_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    infile = datadir("xstatq_08_IN.pkl")
    expfile = datadir("xstatq_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    infile = datadir("xstatq_09_IN.pkl")
    expfile = datadir("xstatq_09_OUT.pkl")
    dummy_test(infile, expfile)


def test_10():
    infile = datadir("xstatq_10_IN.pkl")
    expfile = datadir("xstatq_10_OUT.pkl")
    dummy_test(infile, expfile)


def test_11():
    infile = datadir("xstatq_11_IN.pkl")
    expfile = datadir("xstatq_11_OUT.pkl")
    dummy_test(infile, expfile)


def test_12():
    infile = datadir("xstatq_12_IN.pkl")
    expfile = datadir("xstatq_12_OUT.pkl")
    dummy_test(infile, expfile)


def test_13():
    infile = datadir("statq_13_IN.pkl")
    expfile = datadir("statq_13_OUT.pkl")
    dummy_test(infile, expfile)


def test_14():
    infile = datadir("xstatq_14_IN.pkl")
    expfile = datadir("xstatq_14_OUT.pkl")
    dummy_test(infile, expfile)
