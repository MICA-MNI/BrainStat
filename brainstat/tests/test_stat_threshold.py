import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats.multiple_comparisons import stat_threshold


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

# parameters in *pck is equal to default params, if not specified in tests


def test_01():
    # search_volume = 5.59, df = 5, nconj = 0.5, nprint = 0
    infile  = datadir('thresh_01_IN.pkl')
    expfile = datadir('thresh_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02():
    # search volume is a list,  search volume = [3.2, 8.82, 7.71],
    # df = 5, nconj = 0.5, nprint=0
    infile  = datadir('thresh_02_IN.pkl')
    expfile = datadir('thresh_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03():
    # search volume is a 1D numpy array, search volume = np.array([0.36, 6.22 , 2.13]),
    # df = 5, nconj = 0.5, nprint=0
    infile  = datadir('thresh_03_IN.pkl')
    expfile = datadir('thresh_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04():
    # search_volume is a 2D numpy array,
    # search_volume = array([[ 0.46,  3.00], [13.00, 87.46]]),
    # df = 5, nconj = 0.5, nprint=0
    infile  = datadir('thresh_04_IN.pkl')
    expfile = datadir('thresh_04_OUT.pkl')
    dummy_test(infile, expfile)


def test_05():
    # search_volume is a float,  search_volume = 1.22,
    # num_voxels = 560, df = 5, nconj = 0.5, nprint = 0
    infile  = datadir('thresh_05_IN.pkl')
    expfile = datadir('thresh_05_OUT.pkl')
    dummy_test(infile, expfile)


def test_06():
    # search_volume is a 2D numpy array,
    # search volume = array([[3955,  760], [ 770, 4042]]),
    # num_voxels is a list of integers, num_voxels = [3955, 760, 8058],
    # df = 5, nconj = 0.5, nprint = 0
    infile  = datadir('thresh_06_IN.pkl')
    expfile = datadir('thresh_06_OUT.pkl')
    dummy_test(infile, expfile)


def test_07():
    # search_volume is a 2D numpy array,
    # search_volume = array([[ 49,  47], [ 57, 136]]),
    # num_voxels is a 2D array of shape (23, 1), including floats,
    # fwhm = 5.34, df = 35, nconj = 0.5, nprint=0
    infile  = datadir('thresh_07_IN.pkl')
    expfile = datadir('thresh_07_OUT.pkl')
    dummy_test(infile, expfile)


def test_08():
    # search_volume is a 2D numpy array, array([[ 76,  71], [ 81, 163]]),
    # num_voxels is a 2D array of shape (75, 1), dtype('float64'),
    # fwhm = 3.57,  df = inf, nconj = 0.5, nprint=0
    infile  = datadir('thresh_08_IN.pkl')
    expfile = datadir('thresh_08_OUT.pkl')
    dummy_test(infile, expfile)


def test_09():
    # search_volume is a float,  search_volume = 70.66,
    # num_voxels is a 1D array of shape (5,), dtype('int64'),
    # fwhm = 6.95, df = 5, p_val_peak = 0.71, nconj = 0.5, nprint=0
    infile  = datadir('thresh_09_IN.pkl')
    expfile = datadir('thresh_09_OUT.pkl')
    dummy_test(infile, expfile)


def test_10():
    # search_volume is a float,  search_volume = 72.70,
    # num_voxels is a 1D array of shape (5,), dtype('int64'),
    # fwhm = 5.21, df = 7,
    # p_val_peak is a list,  p_val_peak = [0.48, 0.54, 0.24, 0.27],
    # nconj = 0.5, nprint=0
    infile  = datadir('thresh_10_IN.pkl')
    expfile = datadir('thresh_10_OUT.pkl')
    dummy_test(infile, expfile)


def test_11():
    # search_volume is a float, search_volume = 33.26,
    # num_voxels is a 1D array of shape (5,), dtype('int64'),
    # fwhm = 5.20, df = 7,
    # p_val_peak is a 1D array of shape (7,), dtype('float64'),
    # nconj = 0.5, nprint=0
    infile  = datadir('thresh_11_IN.pkl')
    expfile = datadir('thresh_11_OUT.pkl')
    dummy_test(infile, expfile)


def test_12():
    # search_volume is a float, search_volume = 7.0,
    # num_voxels = 55, fwhm = 9.50, df = 6,
    # p_val_peak is a 1D array of shape (55,), dtype('float64'),
    # cluster_threshold = 0.048,
    # p_val_extent is a 1D array of shape (55,), dtype('float64'),
    # nconj = 54, nprint = 0
    infile  = datadir('thresh_12_IN.pkl')
    expfile = datadir('thresh_12_OUT.pkl')
    dummy_test(infile, expfile)


def test_13():
    # search_volume is a list, search_volume = [5.7, 8, 9],
    # num_voxels = 100, df = 5,
    # p_val_peak is a 2D array of shape (2,2), dtype('float64'),
    # cluster_threshold is a list, cluster_threshold = [0.001, 0.1, 0, 6]
    # p_val_extent = 0.01, nconj = 0.02, nvar = [1, 1], nprint = 0
    infile  = datadir('thresh_13_IN.pkl')
    expfile = datadir('thresh_13_OUT.pkl')
    dummy_test(infile, expfile)


def test_14():
    # search_volume is a 2D numpy array,
    # search_volume = np.array([[4.00e+00, nan, 6.59e+03]]).
    # num_voxels = 64984, fwhm = 1,
    # df a 2D array of shape (2,2), dtype('int64'),
    # p_val_peak is a 1D array of shape (2,), dtype('float64'),
    # cluster_threshold = 0.001, p_val_extent = 0.05, nprint = 1
    infile  = datadir('thresh_14_IN.pkl')
    expfile = datadir('thresh_14_OUT.pkl')
    dummy_test(infile, expfile)


def test_15():
    # df a 2D array of shape (2, 2), dtype('int64'),
    # p_val_peak is a 2D array of shape (1, 87), dtype('float64'),
    # nprint = 0
    infile  = datadir('thresh_15_IN.pkl')
    expfile = datadir('thresh_15_OUT.pkl')
    dummy_test(infile, expfile)


def test_16():
    # df a 2D array of shape (2, 2), dtype('int64'),
    # p_val_peak = 0.001, nvar = 3.0 nprint = 0
    infile  = datadir('thresh_16_IN.pkl')
    expfile = datadir('thresh_16_OUT.pkl')
    dummy_test(infile, expfile)


