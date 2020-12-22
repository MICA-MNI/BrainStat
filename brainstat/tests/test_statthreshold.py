import numpy as np
import pickle
from .testutil import datadir
from ..stats import stat_threshold


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    # run stat_threshold
    A, B, C, D, E, F = stat_threshold(idic["search_volume"],
                                      idic["num_voxels"],
                                      idic["fwhm"],
                                      idic["df"],
                                      idic["p_val_peak"],
                                      idic["cluster_threshold"],
                                      idic["p_val_extent"],
                                      idic["nconj"],
                                      idic["nvar"],
                                      None,
                                      None,
                                      idic["nprint"])
    outdic = {"peak_threshold" : A, "extent_threshold" : B,
              "peak_threshold_1" : C, "extent_threshold_1" : D,
              "t" : E, "rho" : F}

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in outdic.keys():
        comp = np.allclose(outdic[key], expdic[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01(datadir):
    infile  = datadir.join('thresh_01_IN.pkl')
    expfile = datadir.join('thresh_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02(datadir):
    infile  = datadir.join('thresh_02_IN.pkl')
    expfile = datadir.join('thresh_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03(datadir):
    infile  = datadir.join('thresh_03_IN.pkl')
    expfile = datadir.join('thresh_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04(datadir):
    infile  = datadir.join('thresh_04_IN.pkl')
    expfile = datadir.join('thresh_04_OUT.pkl')
    dummy_test(infile, expfile)


def test_05(datadir):
    infile  = datadir.join('thresh_05_IN.pkl')
    expfile = datadir.join('thresh_05_OUT.pkl')
    dummy_test(infile, expfile)


def test_06(datadir):
    infile  = datadir.join('thresh_06_IN.pkl')
    expfile = datadir.join('thresh_06_OUT.pkl')
    dummy_test(infile, expfile)


def test_07(datadir):
    infile  = datadir.join('thresh_07_IN.pkl')
    expfile = datadir.join('thresh_07_OUT.pkl')
    dummy_test(infile, expfile)


def test_08(datadir):
    infile  = datadir.join('thresh_08_IN.pkl')
    expfile = datadir.join('thresh_08_OUT.pkl')
    dummy_test(infile, expfile)


def test_09(datadir):
    infile  = datadir.join('thresh_09_IN.pkl')
    expfile = datadir.join('thresh_09_OUT.pkl')
    dummy_test(infile, expfile)


def test_10(datadir):
    infile  = datadir.join('thresh_10_IN.pkl')
    expfile = datadir.join('thresh_10_OUT.pkl')
    dummy_test(infile, expfile)


def test_11(datadir):
    infile  = datadir.join('thresh_11_IN.pkl')
    expfile = datadir.join('thresh_11_OUT.pkl')
    dummy_test(infile, expfile)


def test_12(datadir):
    infile  = datadir.join('thresh_12_IN.pkl')
    expfile = datadir.join('thresh_12_OUT.pkl')
    dummy_test(infile, expfile)


def test_13(datadir):
    infile  = datadir.join('thresh_13_IN.pkl')
    expfile = datadir.join('thresh_13_OUT.pkl')
    dummy_test(infile, expfile)


def test_14(datadir):
    infile  = datadir.join('thresh_14_IN.pkl')
    expfile = datadir.join('thresh_14_OUT.pkl')
    dummy_test(infile, expfile)


def test_15(datadir):
    infile  = datadir.join('thresh_15_IN.pkl')
    expfile = datadir.join('thresh_15_OUT.pkl')
    dummy_test(infile, expfile)


def test_16(datadir):
    infile  = datadir.join('thresh_16_IN.pkl')
    expfile = datadir.join('thresh_16_OUT.pkl')
    dummy_test(infile, expfile)




