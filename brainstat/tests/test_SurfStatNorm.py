import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatNorm


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

    # run SurfStatNorm
    Y_out, Yav_out = SurfStatNorm(Y, mask, subdiv)

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
    infile  = datadir('statnor_01_IN.pkl')
    expfile = datadir('statnor_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02():
    infile  = datadir('statnor_02_IN.pkl')
    expfile = datadir('statnor_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03():
    infile  = datadir('statnor_03_IN.pkl')
    expfile = datadir('statnor_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04():
    infile  = datadir('statnor_04_IN.pkl')
    expfile = datadir('statnor_04_OUT.pkl')
    dummy_test(infile, expfile)


