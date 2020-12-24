import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatSmooth


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y = idic['Y']
    FWHM = idic['FWHM']

    surf = {}
    if 'tri' in idic.keys():
        surf['tri'] = idic['tri']

    if 'lat' in idic.keys():
        surf['lat'] = idic['lat']

    # run SurfStatSmooth
    Y_out = SurfStatSmooth(Y, surf, FWHM)

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()
    Y_exp = expdic['Python_Y']

    testout = []

    comp = np.allclose(Y_out, Y_exp, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    # ['Y'] : np array, shape (72, 72), float64
    # ['tri'] : np array, shape (1, 3), int64
    # ['FWHM'] : float
    infile  = datadir('statsmo_01_IN.pkl')
    expfile = datadir('statsmo_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02():
    # ['Y'] : np array, shape (44, 44), float64
    # ['tri'] : np array, shape (50, 3), int64
    # ['FWHM'] : float
    infile  = datadir('statsmo_02_IN.pkl')
    expfile = datadir('statsmo_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03():
    # ['Y'] : np array, shape (94, 94), float64
    # ['lat'] : np array, shape (3, 3, 3), int64
    # ['FWHM'] : float
    infile  = datadir('statsmo_03_IN.pkl')
    expfile = datadir('statsmo_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04():
    # ['Y'] : np array, shape (68, 3, 2), float64
    # ['tri'] : np array, shape (1, 3), int64
    # ['FWHM'] : float
    infile  = datadir('statsmo_04_IN.pkl')
    expfile = datadir('statsmo_04_OUT.pkl')
    dummy_test(infile, expfile)


def test_05():
    # ['Y'] : np array, shape (1, 1024), float64
    # ['tri'] : np array, shape (2044, 3), uint16
    # ['FWHM'] : float
    infile  = datadir('statsmo_05_IN.pkl')
    expfile = datadir('statsmo_05_OUT.pkl')
    dummy_test(infile, expfile)


def test_06():
    # ['Y'] : np array, shape (91, 1024), float64
    # ['tri'] : np array, shape (2044, 3), uint16
    # ['FWHM'] : float
    infile  = datadir('statsmo_06_IN.pkl')
    expfile = datadir('statsmo_06_OUT.pkl')
    dummy_test(infile, expfile)


def test_07():
    # ['Y'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['FWHM'] : int
    infile  = datadir('statsmo_07_IN.pkl')
    expfile = datadir('statsmo_07_OUT.pkl')
    dummy_test(infile, expfile)


def test_08():
    # ['Y'] : np array, shape (26, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['FWHM'] : int
    infile  = datadir('statsmo_08_IN.pkl')
    expfile = datadir('statsmo_08_OUT.pkl')
    dummy_test(infile, expfile)


def test_09():
    # ['Y'] : np array, shape (1, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['FWHM'] : int
    infile  = datadir('statsmo_09_IN.pkl')
    expfile = datadir('statsmo_09_OUT.pkl')
    dummy_test(infile, expfile)


def test_10():
    # ['Y'] : np array, shape (60, 20484), float64
    # ['tri'] : np array, shape (40960, 3), int32
    # ['FWHM'] : float
    infile  = datadir('statsmo_10_IN.pkl')
    expfile = datadir('statsmo_10_OUT.pkl')
    dummy_test(infile, expfile)


