import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatStand
import gzip

def dummy_test(infile, expfile):

    # load input test data
    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    Y = idic['Y']

    mask = None
    subtractordivide = 's'


    if 'mask' in idic.keys():
        mask = idic['mask']

    if 'subtractordivide' in idic.keys():
        subtractordivide = idic['subtractordivide']


    # run SurfStatStand
    Y_out, Ym_out = SurfStatStand(Y, mask, subtractordivide)

    # load expected outout data
    with gzip.open(expfile, 'rb') as f:
        expdic  = pickle.load(f)
    Y_exp  = expdic['Python_Y']
    Ym_exp = expdic['Python_Ym']

    testout = []

    testout.append(np.allclose(Y_out, Y_exp, rtol=1e-05, equal_nan=True))
    testout.append(np.allclose(Ym_out, Ym_exp, rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = datadir('statsta_01_IN.pkl.gz')
    expfile = datadir('statsta_01_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_02():
    infile  = datadir('statsta_02_IN.pkl.gz')
    expfile = datadir('statsta_02_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_03():
    infile  = datadir('statsta_03_IN.pkl.gz')
    expfile = datadir('statsta_03_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_04():
    infile  = datadir('statsta_04_IN.pkl.gz')
    expfile = datadir('statsta_04_OUT.pkl.gz')
    dummy_test(infile, expfile)


