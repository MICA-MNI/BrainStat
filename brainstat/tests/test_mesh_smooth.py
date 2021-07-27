"""Unit tests of mesh_smooth."""
import pickle

import numpy as np

from brainstat.mesh.data import mesh_smooth

from .testutil import datadir


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    Y = idic["Y"]
    FWHM = idic["FWHM"]

    surf = {}
    if "tri" in idic.keys():
        surf["tri"] = idic["tri"]

    if "lat" in idic.keys():
        surf["lat"] = idic["lat"]

    # run mesh_smooth
    Y_out = mesh_smooth(Y, surf, FWHM)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()
    Y_exp = expdic["Python_Y"]

    testout = []

    comp = np.allclose(Y_out, Y_exp, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    infile = datadir("xstatsmo_01_IN.pkl")
    expfile = datadir("xstatsmo_01_OUT.pkl")
    dummy_test(infile, expfile)
