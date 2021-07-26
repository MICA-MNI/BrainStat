"""Unit tests of mesh_standardize."""
import pickle

import numpy as np
import pytest

from brainstat.mesh.data import mesh_standardize

from .testutil import datadir


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    Y = idic["Y"]

    mask = None
    subtractordivide = "s"

    if "mask" in idic.keys():
        mask = idic["mask"]

    if "subtractordivide" in idic.keys():
        subtractordivide = idic["subtractordivide"]

    # run mesh_standardize
    Y_out, Ym_out = mesh_standardize(Y, mask, subtractordivide)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()
    Y_exp = expdic["Python_Y"]
    Ym_exp = expdic["Python_Ym"]

    testout = []

    testout.append(np.allclose(Y_out, Y_exp, rtol=1e-05, equal_nan=True))
    testout.append(np.allclose(Ym_out, Ym_exp, rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout)


@pytest.mark.skip(reason="Function pending deprecation.")
def test_01():
    # ['Y'] : np array, shape (1, 1), int64
    infile = datadir("statsta_01_IN.pkl")
    expfile = datadir("statsta_01_OUT.pkl")
    dummy_test(infile, expfile)


@pytest.mark.skip(reason="Function pending deprecation.")
def test_02():
    # ['Y'] : np array, shape (1, 10), int64
    # ['mask'] : np array, shape (10,), bool
    infile = datadir("statsta_02_IN.pkl")
    expfile = datadir("statsta_02_OUT.pkl")
    dummy_test(infile, expfile)


@pytest.mark.skip(reason="Function pending deprecation.")
def test_03():
    # ['Y'] : np array, shape (2, 10), int64
    # ['mask'] : np array, shape (10,), bool
    infile = datadir("statsta_03_IN.pkl")
    expfile = datadir("statsta_03_OUT.pkl")
    dummy_test(infile, expfile)


@pytest.mark.skip(reason="Function pending deprecation.")
def test_04():
    # ['Y'] : np array, shape (3, 4, 2), float64
    # ['mask'] : np array, shape (4,), bool
    infile = datadir("statsta_04_IN.pkl")
    expfile = datadir("statsta_04_OUT.pkl")
    dummy_test(infile, expfile)
