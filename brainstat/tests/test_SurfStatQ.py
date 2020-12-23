import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatQ
import gzip

def dummy_test(infile, expfile):

    # load input test data
    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    slm = {}
    slm['t']  = idic['t']
    slm['df'] = idic['df']
    slm['k']  = idic['k']

    # check other potential keys in input
    if 'tri' in idic.keys():
        slm['tri']    = idic['tri']

    if 'resl' in idic.keys():
        slm['resl'] = idic['resl']

    if 'dfs' in idic.keys():
        slm['dfs'] = idic['dfs']

    if 'du' in idic.keys():
        slm['du'] = idic['du']

    if 'c' in idic.keys():
        slm['c']    = idic['c']

    if 'k' in idic.keys():
        slm['k']    = idic['k']

    if 'ef' in idic.keys():
        slm['ef']    = idic['ef']

    if 'sd' in idic.keys():
        slm['sd']    = idic['sd']

    if 't' in idic.keys():
        slm['t']    = idic['t']

    if 'SSE' in idic.keys():
        slm['SSE']    = idic['SSE']

    if 'mask' in idic.keys():
        mask = idic['mask']
    else:
        mask = None


    # run SurfStatQ
    outdic = SurfStatQ(slm, mask)

    # load expected outout data
    with gzip.open(expfile, 'rb') as f:
        expdic  = pickle.load(f)

    testout = []

    for key in outdic.keys():
        comp = np.allclose(outdic[key], expdic[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = datadir('statq_01_IN.pkl.gz')
    expfile = datadir('statq_01_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_02():
    infile  = datadir('statq_02_IN.pkl.gz')
    expfile = datadir('statq_02_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_03():
    infile  = datadir('statq_03_IN.pkl.gz')
    expfile = datadir('statq_03_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_04():
    infile  = datadir('statq_04_IN.pkl.gz')
    expfile = datadir('statq_04_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_05():
    infile  = datadir('statq_05_IN.pkl.gz')
    expfile = datadir('statq_05_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_06():
    infile  = datadir('statq_06_IN.pkl.gz')
    expfile = datadir('statq_06_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_07():
    infile  = datadir('statq_07_IN.pkl.gz')
    expfile = datadir('statq_07_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_08():
    infile  = datadir('statq_08_IN.pkl.gz')
    expfile = datadir('statq_08_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_09():
    infile  = datadir('statq_09_IN.pkl.gz')
    expfile = datadir('statq_09_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_10():
    infile  = datadir('statq_10_IN.pkl.gz')
    expfile = datadir('statq_10_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_11():
    infile  = datadir('statq_11_IN.pkl.gz')
    expfile = datadir('statq_11_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_12():
    infile  = datadir('statq_12_IN.pkl.gz')
    expfile = datadir('statq_12_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_13():
    infile  = datadir('statq_13_IN.pkl.gz')
    expfile = datadir('statq_13_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_14():
    infile  = datadir('statq_14_IN.pkl.gz')
    expfile = datadir('statq_14_OUT.pkl.gz')
    dummy_test(infile, expfile)









