import numpy as np
import pickle
from .testutil import datadir
from brainstat.mesh.utils import mesh_edges


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
    # ['tri'] is a 2D numpy array of shape (78, 3), dtype('float64')
    infile = datadir("xstatedg_01_IN.pkl")
    expfile = datadir("xstatedg_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    # ['lat'] is a 2D numpy array of shape (10, 10), dtype('float64')
    infile = datadir("xstatedg_02_IN.pkl")
    expfile = datadir("xstatedg_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    # ['lat'] is a 3D numpy array of shape (10, 10, 10), dtype('int64')
    infile = datadir("xstatedg_03_IN.pkl")
    expfile = datadir("xstatedg_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    # ['tri'] is a 2D numpy array of shape (2044, 3), dtype('uint16')
    infile = datadir("xstatedg_04_IN.pkl")
    expfile = datadir("xstatedg_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    # ['tri'] is a 2D numpy array of shape (40960, 3), from "thickness_n10.pkl"
    infile = datadir("thickness_n10.pkl")
    expfile = datadir("xstatedg_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    # test 05 thickiness_n10 data is shuffled
    infile = datadir("xstatedg_06_IN.pkl")
    expfile = datadir("xstatedg_06_OUT.pkl")
    dummy_test(infile, expfile)
