import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatT
import gzip

def dummy_test(infile, expfile):

    # load input test data
    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    slm = {}
    slm['X']    = idic['X']
    slm['df']   = idic['df']
    slm['coef'] = idic['coef']
    slm['SSE']  = idic['SSE']

    contrast = idic['contrast']

    if 'V' in idic.keys():
        slm['V']    = idic['V']

    if 'SSE' in idic.keys():
        slm['SSE']    = idic['SSE']

    if 'r' in idic.keys():
        slm['r']    = idic['r']

    if 'dr' in idic.keys():
        slm['dr']    = idic['dr']

    if 'tri' in idic.keys():
        slm['tri']    = idic['tri']

    if 'resl' in idic.keys():
        slm['resl']    = idic['resl']


    # run SurfStatT
    outdic = SurfStatT(slm, contrast)

    # load expected outout data
    with gzip.open(expfile, 'rb') as f:
        expdic  = pickle.load(f)

    testout = []

    for key in outdic.keys():
        comp = np.allclose(outdic[key], expdic[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = datadir('statt_01_IN.pkl.gz')
    expfile = datadir('statt_01_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_02():
    infile  = datadir('statt_02_IN.pkl.gz')
    expfile = datadir('statt_02_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_03():
    infile  = datadir('statt_03_IN.pkl.gz')
    expfile = datadir('statt_03_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_04():
    infile  = datadir('statt_04_IN.pkl.gz')
    expfile = datadir('statt_04_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_05():
    infile  = datadir('statt_05_IN.pkl.gz')
    expfile = datadir('statt_05_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_06():
    infile  = datadir('statt_06_IN.pkl.gz')
    expfile = datadir('statt_06_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_07():
    infile  = datadir('statt_07_IN.pkl.gz')
    expfile = datadir('statt_07_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_08():
    infile  = datadir('statt_08_IN.pkl.gz')
    expfile = datadir('statt_08_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_09():
    infile  = datadir('statt_09_IN.pkl.gz')
    expfile = datadir('statt_09_OUT.pkl.gz')
    dummy_test(infile, expfile)


