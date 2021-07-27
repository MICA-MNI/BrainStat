"""Unit tests of mesh_normalize."""
import pickle

import numpy as np

from brainstat.mesh.data import mesh_normalize

from .testutil import datadir


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    Y = idic["Y"]

    mask = None
    subdiv = "s"

    if "mask" in idic.keys():
        mask = idic["mask"]

    # run mesh_normalize
    Y_out, Yav_out = mesh_normalize(Y, mask, subdiv)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()
    exp_Y_out = expdic["Python_Y"]
    exp_Yav_out = expdic["Python_Yav"]

    testout = []
    testout.append(np.allclose(Y_out, exp_Y_out, rtol=1e-05, equal_nan=True))
    testout.append(np.allclose(Yav_out, exp_Yav_out, rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout)


def test_01():
    infile = datadir("xstatnor_01_IN.pkl")
    expfile = datadir("xstatnor_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    infile = datadir("xstatnor_02_IN.pkl")
    expfile = datadir("xstatnor_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    infile = datadir("xstatnor_03_IN.pkl")
    expfile = datadir("xstatnor_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    infile = datadir("xstatnor_04_IN.pkl")
    expfile = datadir("xstatnor_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    infile = datadir("xstatnor_05_IN.pkl")
    expfile = datadir("xstatnor_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    infile = datadir("xstatnor_06_IN.pkl")
    expfile = datadir("xstatnor_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    infile = datadir("xstatnor_07_IN.pkl")
    expfile = datadir("xstatnor_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    infile = datadir("xstatnor_08_IN.pkl")
    expfile = datadir("xstatnor_08_OUT.pkl")
    dummy_test(infile, expfile)
