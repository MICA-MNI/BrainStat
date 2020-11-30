import sys
sys.path.append("python")
from SurfStatNorm import *
import numpy as np
import pytest
import pickle


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y = idic['Y']

    mask = None
    subdiv = 's'

    if 'mask' in idic.keys():
        mask = idic['mask']

    if 's' in idic.keys():
        s = idic['s']

    # run SurfStatNorm
    Y_out, Yav_out = py_SurfStatNorm(Y, mask, subdiv)

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()
    exp_Y_out = expdic['Python_Y']
    exp_Yav_out = expdic['Python_Yav']

    testout = []
    testout.append(np.allclose(Y_out, exp_Y_out, rtol=1e-05, equal_nan=True))
    testout.append(np.allclose(Yav_out, exp_Yav_out, rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = './tests/data/unitdata/statnor_01_IN.pkl'
    expfile = './tests/data/unitdata/statnor_01_OUT.pkl'
    dummy_test(infile, expfile)


def test_02():
    infile  = './tests/data/unitdata/statnor_02_IN.pkl'
    expfile = './tests/data/unitdata/statnor_02_OUT.pkl'
    dummy_test(infile, expfile)


def test_03():
    infile  = './tests/data/unitdata/statnor_03_IN.pkl'
    expfile = './tests/data/unitdata/statnor_03_OUT.pkl'
    dummy_test(infile, expfile)


def test_04():
    infile  = './tests/data/unitdata/statnor_04_IN.pkl'
    expfile = './tests/data/unitdata/statnor_04_OUT.pkl'
    dummy_test(infile, expfile)

