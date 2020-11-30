import sys
sys.path.append("python")
from SurfStatStand import *
import numpy as np
import pickle
import pytest


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y = idic['Y']

    mask = None
    subtractordivide = 's'


    if 'mask' in idic.keys():
        mask = idic['mask']

    if 'subtractordivide' in idic.keys():
        subtractordivide = idic['subtractordivide']


    # run SurfStatStand
    Y_out, Ym_out = py_SurfStatStand(Y, mask, subtractordivide)

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()
    Y_exp  = expdic['Python_Y']
    Ym_exp = expdic['Python_Ym']

    testout = []

    testout.append(np.allclose(Y_out, Y_exp, rtol=1e-05, equal_nan=True))
    testout.append(np.allclose(Ym_out, Ym_exp, rtol=1e-05, equal_nan=True))


    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = './tests/data/unitdata/statsta_01_IN.pkl'
    expfile = './tests/data/unitdata/statsta_01_OUT.pkl'
    dummy_test(infile, expfile)


def test_02():
    infile  = './tests/data/unitdata/statsta_02_IN.pkl'
    expfile = './tests/data/unitdata/statsta_02_OUT.pkl'
    dummy_test(infile, expfile)


def test_03():
    infile  = './tests/data/unitdata/statsta_03_IN.pkl'
    expfile = './tests/data/unitdata/statsta_03_OUT.pkl'
    dummy_test(infile, expfile)


def test_04():
    infile  = './tests/data/unitdata/statsta_04_IN.pkl'
    expfile = './tests/data/unitdata/statsta_04_OUT.pkl'
    dummy_test(infile, expfile)


