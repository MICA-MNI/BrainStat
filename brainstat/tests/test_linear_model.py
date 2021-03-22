import numpy as np
from .testutil import datadir
from brainstat.stats.terms import Term
from brainstat.stats.SLM import SLM
import pickle


def dummy_test(infile, expfile, simple=True):

    ifile = open(infile, "br")
    Din = pickle.load(ifile)
    ifile.close()

    Y = Din["Y"]
    M = Din["M"]

    # assign slm params
    slm = SLM(M, Term(1))

    if "tri" in Din:
        slm.surf = {"tri": Din["tri"]}
    if "lat" in Din:
        slm.surf = {"lat": Din["lat"]}

    # here we go --> run the linear model
    slm.linear_model(Y)

    ofile = open(expfile, "br")
    Dout = pickle.load(ofile)
    ofile.close()

    # compare...
    testout = []
    for makey_ in Dout.keys():
        comp = np.allclose(
            getattr(slm, makey_), Dout[makey_], rtol=1e-05, equal_nan=True
        )
        testout.append(comp)
    assert all(flag == True for (flag) in testout)


def test_01():
    # ["Y"] and ["M"] small 2D suqare arrays
    # ["Y"] : np array, shape (43, 43), dtype("float64")
    # ["M"] : np array, (43, 43), dtype("float64")
    infile = datadir("xlinmod_01_IN.pkl")
    expfile = datadir("xlinmod_01_OUT.pkl")
    dummy_test(infile, expfile)


def test_02():
    # ["Y"] and ["M"] small 2D rectengular arrays
    # ["Y"] : np array, (62, 7), dtype("float64")
    # ["M"] : np array, (62, 92), dtype("float64")
    infile = datadir("xlinmod_02_IN.pkl")
    expfile = datadir("xlinmod_02_OUT.pkl")
    dummy_test(infile, expfile)


def test_03():
    # ["Y"] is a 3D array, ["M"] is a 2D array
    # ["Y"] : np array, (52, 64, 76), dtype("float64")
    # ["M"] : np array, (52, 2), dtype("float64")
    infile = datadir("xlinmod_03_IN.pkl")
    expfile = datadir("xlinmod_03_OUT.pkl")
    dummy_test(infile, expfile)


def test_04():
    # similar to test_03, shapes of ["Y"] and ["M"] changed
    # ["Y"] : np array, (69, 41, 5), dtype("float64")
    # ["M"] : np array, (69, 30), dtype("float64")
    infile = datadir("xlinmod_04_IN.pkl")
    expfile = datadir("xlinmod_04_OUT.pkl")
    dummy_test(infile, expfile)


def test_05():
    # ["Y"] and ["M"] small 2D rectengular arrays, size(Y) < size(M)
    # ["Y"] : np array, (81, 1), dtype("float64")
    # ["M"] : np array, (81, 2), dtype("float64")
    infile = datadir("xlinmod_05_IN.pkl")
    expfile = datadir("xlinmod_05_OUT.pkl")
    dummy_test(infile, expfile)


def test_06():
    # ["Y"] is a 3D array, ["M"] is a 2D array, M has more columns than Y
    # ["Y"] : np array, (93, 41, 57), dtype("float64")
    # ["M"] : np array, (93, 67), dtype("float64")
    infile = datadir("xlinmod_06_IN.pkl")
    expfile = datadir("xlinmod_06_OUT.pkl")
    dummy_test(infile, expfile)


def test_07():
    # similar to test_06, differently shaped Y and M
    # ["Y"] : np array, (40, 46, 21), dtype("float64")
    # ["M"] : np array, (40, 81), dtype("float64")
    infile = datadir("xlinmod_07_IN.pkl")
    expfile = datadir("xlinmod_07_OUT.pkl")
    dummy_test(infile, expfile)


def test_08():
    # ["Y"] and ["M"] mid. sized 2D arrays + optional ["tri"] input for surf
    # ["Y"] : np array, (93, 43), dtype("float64")
    # ["M"] : np array, (93, 2), dtype("float64")
    # ["tri"] : np array, (93, 3), dtype("int64")
    infile = datadir("xlinmod_08_IN.pkl")
    expfile = datadir("xlinmod_08_OUT.pkl")
    dummy_test(infile, expfile)


def test_09():
    # ["Y"] is 3D array, ["M"] is 2D array, M has more cols than Y, tri given
    # ["Y"] : np array, (98, 69, 60), dtype("float64")
    # ["M"] : np array, (98, 91), dtype("float64")
    # ["tri"] : np array, (60, 3), dtype("int64")
    infile = datadir("xlinmod_09_IN.pkl")
    expfile = datadir("xlinmod_09_OUT.pkl")
    dummy_test(infile, expfile)


def test_10():
    # similar to test_02 + optional ["lat"] input for surf
    # ["Y"] : np array, (49, 27), dtype("float64")
    # ["M"] : np array, (49, 2), dtype("float64")
    # ["lat"] : np array, (3, 3, 3), dtype("int64"), 1"s or 0"s
    infile = datadir("xlinmod_10_IN.pkl")
    expfile = datadir("xlinmod_10_OUT.pkl")
    dummy_test(infile, expfile)


def test_11():
    # similar to test_03 + optional ["lat"] input for surf
    # ["Y"] : np array, (45, 27, 3), dtype("float64")
    # ["M"] : np array, (45, 7), dtype("float64")
    # ["lat"] : np array, (3, 3, 3), dtype("int64"), 1"s or 0"s
    infile = datadir("xlinmod_11_IN.pkl")
    expfile = datadir("xlinmod_11_OUT.pkl")
    dummy_test(infile, expfile)


def test_12():
    # real dataset, ["Y"] 20k columns, ["M"] ages, ["tri"] 40k vertex
    # ["Y"] : np array, (10, 20484), dtype("float64")
    # ["M"] : np array, (1, 10), dtype("float64")
    # ["tri"] : np array, (40960, 3), dtype("int32")
    infile = datadir("thickness_n10.pkl")
    expfile = datadir("xlinmod_12_OUT.pkl")
    dummy_test(infile, expfile, simple=False)


def test_13():
    # similar to test_12, ["Y"] values shuffled
    # ["Y"] : np array, (10, 20484), dtype("float64")
    # ["M"] : np array, (1, 10), dtype("float64")
    # ["tri"] : np array, (40960, 3), dtype("int32")
    infile = datadir("xlinmod_13_IN.pkl")
    expfile = datadir("xlinmod_13_OUT.pkl")
    dummy_test(infile, expfile, simple=False)


def test_14():
    # similar to test_12, ["Y"] and ["tri"] values shuffled
    # ["Y"] : np array, (10, 20484), dtype("float64")
    # ["M"] : np array, (1, 10), dtype("float64")
    # ["tri"] : np array, (40960, 3), dtype("int32")
    infile = datadir("xlinmod_14_IN.pkl")
    expfile = datadir("xlinmod_14_OUT.pkl")
    dummy_test(infile, expfile, simple=False)


def test_15():
    # similar to test_12, ["Y"] size doubled + model params extended
    # ["Y"] : np array, (20, 20484), dtype("float64")
    # ["M"] : np array, (20, 9), dtype("uint16")
    # ["tri"] : np array, (40960, 3), dtype("int32")
    infile = datadir("xlinmod_15_IN.pkl")
    expfile = datadir("xlinmod_15_OUT.pkl")
    dummy_test(infile, expfile)

