import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatF
import gzip


def dummy_test(infile, expfile):

    # load input test data
    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    slm1 = {}
    slm1['X']    = idic['slm1X']
    slm1['df']   = idic['slm1df']
    slm1['SSE']  = idic['slm1SSE']
    slm1['coef'] = idic['slm1coef']

    slm2 = {}
    slm2['X']    = idic['slm2X']
    slm2['df']   = idic['slm2df']
    slm2['SSE']  = idic['slm2SSE']
    slm2['coef'] = idic['slm2coef']

    # check other potential keys in input
    if 'slm1tri' in idic.keys():
        slm1['tri']    = idic['slm1tri']
        slm2['tri']    = idic['slm2tri']

    if 'slm1resl' in idic.keys():
        slm1['resl'] = idic['slm1resl']
        slm2['resl'] = idic['slm2resl']

    if 'slm1c' in idic.keys():
        slm1['c']    = idic['slm1c']
        slm2['c']    = idic['slm2c']

    if 'slm1k' in idic.keys():
        slm1['k']    = idic['slm1k']
        slm2['k']    = idic['slm2k']

    if 'slm1ef' in idic.keys():
        slm1['ef']    = idic['slm1ef']
        slm2['ef']    = idic['slm2ef']

    if 'slm1sd' in idic.keys():
        slm1['sd']    = idic['slm1sd']
        slm2['sd']    = idic['slm2sd']

    if 'slm1t' in idic.keys():
        slm1['t']    = idic['slm1t']
        slm2['t']    = idic['slm2t']



    # run SurfStatF
    outdic = SurfStatF(slm1, slm2)

    # load expected outout data
    with gzip.open(expfile, 'rb') as f:
        expdic  = pickle.load(f)

    testout = []

    for key in outdic.keys():
        comp = np.allclose(outdic[key], expdic[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = datadir('statf_01_IN.pkl.gz')
    expfile = datadir('statf_01_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_02():
    infile  = datadir('statf_02_IN.pkl.gz')
    expfile = datadir('statf_02_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_03():
    infile  = datadir('statf_03_IN.pkl.gz')
    expfile = datadir('statf_03_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_04():
    infile  = datadir('statf_04_IN.pkl.gz')
    expfile = datadir('statf_04_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_05():
    infile  = datadir('statf_05_IN.pkl.gz')
    expfile = datadir('statf_05_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_06():
    infile  = datadir('statf_06_IN.pkl.gz')
    expfile = datadir('statf_06_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_07():
    infile  = datadir('statf_07_IN.pkl.gz')
    expfile = datadir('statf_07_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_08():
    infile  = datadir('statf_08_IN.pkl.gz')
    expfile = datadir('statf_08_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_09():
    infile  = datadir('statf_09_IN.pkl.gz')
    expfile = datadir('statf_09_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_10():
    infile  = datadir('statf_10_IN.pkl.gz')
    expfile = datadir('statf_10_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_11():
    infile  = datadir('statf_11_IN.pkl.gz')
    expfile = datadir('statf_11_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_12():
    infile  = datadir('statf_12_IN.pkl.gz')
    expfile = datadir('statf_12_OUT.pkl.gz')
    dummy_test(infile, expfile)


