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
    infile  = datadir('statsmo_01_IN.pkl')
    expfile = datadir('statsmo_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02():
    infile  = datadir('statsmo_02_IN.pkl')
    expfile = datadir('statsmo_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03():
    infile  = datadir('statsmo_03_IN.pkl')
    expfile = datadir('statsmo_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04():
    infile  = datadir('statsmo_04_IN.pkl')
    expfile = datadir('statsmo_04_OUT.pkl')
    dummy_test(infile, expfile)


def test_05():
    infile  = datadir('statsmo_05_IN.pkl')
    expfile = datadir('statsmo_05_OUT.pkl')
    dummy_test(infile, expfile)


def test_06():
    infile  = datadir('statsmo_06_IN.pkl')
    expfile = datadir('statsmo_06_OUT.pkl')
    dummy_test(infile, expfile)


def test_07():
    infile  = datadir('statsmo_07_IN.pkl')
    expfile = datadir('statsmo_07_OUT.pkl')
    dummy_test(infile, expfile)


def test_08():
    infile  = datadir('statsmo_08_IN.pkl')
    expfile = datadir('statsmo_08_OUT.pkl')
    dummy_test(infile, expfile)


def test_09():
    infile  = datadir('statsmo_09_IN.pkl')
    expfile = datadir('statsmo_09_OUT.pkl')
    dummy_test(infile, expfile)


def test_10():
    infile  = datadir('statsmo_10_IN.pkl')
    expfile = datadir('statsmo_10_OUT.pkl')
    dummy_test(infile, expfile)


