"""Unit tests of mesh_edges."""

import pickle

import numpy as np

from brainstat.mesh.utils import mesh_edges

from .testutil import datadir


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()

    surf = {}

    if "tri" in idic.keys():
        surf["tri"] = idic["tri"]

    if "lat" in idic.keys():
        surf["lat"] = idic["lat"]

    # run mesh_edges
    out_edge = mesh_edges(surf)

    # load expected outout data
    efile = open(expfile, "br")
    expdic = pickle.load(efile)
    efile.close()
    exp_edge = expdic["edg"]

    testout = []

    comp = np.allclose(out_edge, exp_edge, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    assert all(flag == True for (flag) in testout)


# data *pkl consists of either keys ['tri'] or ['lat'], which will be assigned to
# the surf{} dictionary while testing
def test_01():
    infile = datadir("xstatedg_01_IN.pkl")
    expfile = datadir("xstatedg_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    infile = datadir("xstatedg_02_IN.pkl")
    expfile = datadir("xstatedg_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    infile = datadir("xstatedg_03_IN.pkl")
    expfile = datadir("xstatedg_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    infile = datadir("xstatedg_04_IN.pkl")
    expfile = datadir("xstatedg_04_OUT.pkl")
    dummy_test(infile, expfile)
