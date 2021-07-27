"""Unit tests of f-test."""

import pickle

import numpy as np

from brainstat.stats.SLM import SLM, f_test
from brainstat.stats.terms import FixedEffect

from .testutil import datadir


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    slm1 = SLM(FixedEffect(1), FixedEffect(1))
    slm2 = SLM(FixedEffect(1), FixedEffect(2))
    for key in idic.keys():
        if "1" in key:
            setattr(slm1, key[4:], idic[key])
        elif "2" in key:
            setattr(slm2, key[4:], idic[key])

    # run f test
    outdic = f_test(slm1, slm2)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in expdic.keys():
        comp = np.allclose(
            getattr(outdic, key), expdic[key], rtol=1e-05, equal_nan=True
        )
        testout.append(comp)

    assert all(flag == True for (flag) in testout)


# test data *pkl consists of slm1* and slm2* keys
# slm1* variables will be assigned to slm1 dictionary, and slm2* to the slm2 dict.


def test_01():
    infile = datadir("xstatf_01_IN.pkl")
    expfile = datadir("xstatf_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    infile = datadir("xstatf_02_IN.pkl")
    expfile = datadir("xstatf_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    infile = datadir("xstatf_03_IN.pkl")
    expfile = datadir("xstatf_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    infile = datadir("xstatf_04_IN.pkl")
    expfile = datadir("xstatf_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    infile = datadir("xstatf_05_IN.pkl")
    expfile = datadir("xstatf_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    infile = datadir("xstatf_06_IN.pkl")
    expfile = datadir("xstatf_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    infile = datadir("xstatf_07_IN.pkl")
    expfile = datadir("xstatf_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    infile = datadir("xstatf_08_IN.pkl")
    expfile = datadir("xstatf_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    infile = datadir("xstatf_09_IN.pkl")
    expfile = datadir("xstatf_09_OUT.pkl")
    dummy_test(infile, expfile)


def test_10():
    infile = datadir("xstatf_10_IN.pkl")
    expfile = datadir("xstatf_10_OUT.pkl")
    dummy_test(infile, expfile)


def test_11():
    infile = datadir("xstatf_11_IN.pkl")
    expfile = datadir("xstatf_11_OUT.pkl")
    dummy_test(infile, expfile)


def test_12():
    infile = datadir("xstatf_12_IN.pkl")
    expfile = datadir("xstatf_12_OUT.pkl")
    dummy_test(infile, expfile)


def test_13():
    infile = datadir("xstatf_13_IN.pkl")
    expfile = datadir("xstatf_13_OUT.pkl")
    dummy_test(infile, expfile)


def test_14():
    infile = datadir("xstatf_14_IN.pkl")
    expfile = datadir("xstatf_14_OUT.pkl")
    dummy_test(infile, expfile)


def test_15():
    infile = datadir("xstatf_15_IN.pkl")
    expfile = datadir("xstatf_15_OUT.pkl")
    dummy_test(infile, expfile)


def test_16():
    infile = datadir("xstatf_16_IN.pkl")
    expfile = datadir("xstatf_16_OUT.pkl")
    dummy_test(infile, expfile)
