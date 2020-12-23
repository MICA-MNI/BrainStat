import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatResels
import gzip


def dummy_test(infile, expfile):

    # load input test data
    with gzip.open(infile, 'rb') as f:
        idic = pickle.load(f)

    slm = {}

    if 'tri' in idic.keys():
        slm['tri'] = idic['tri']

    if 'resl' in idic.keys():
        slm['resl'] = idic['resl']

    if 'lat' in idic.keys():
        slm['lat'] = idic['lat']

    mask = None

    if 'mask' in idic.keys():
        mask = idic['mask']

    resels_py, reselspvert_py, edg_py = SurfStatResels(slm, mask)

    out = {}
    out['resels'] = resels_py
    out['reselspvert'] = reselspvert_py
    out['edg'] = edg_py

    # load expected outout data
    with gzip.open(expfile, 'rb') as f:
        expdic = pickle.load(f)

    testout = []

    for key in out.keys():
        if out[key] is not None and expdic[key] is not None:
            comp = np.allclose(out[key], expdic[key],
                               rtol=1e-05, equal_nan=True)
            testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    infile = datadir('statresl_01_IN.pkl.gz')
    expfile = datadir('statresl_01_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_02():
    infile = datadir('statresl_02_IN.pkl.gz')
    expfile = datadir('statresl_02_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_03():
    infile = datadir('statresl_03_IN.pkl.gz')
    expfile = datadir('statresl_03_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_04():
    infile = datadir('statresl_04_IN.pkl.gz')
    expfile = datadir('statresl_04_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_05():
    infile = datadir('statresl_05_IN.pkl.gz')
    expfile = datadir('statresl_05_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_06():
    infile = datadir('statresl_06_IN.pkl.gz')
    expfile = datadir('statresl_06_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_07():
    infile = datadir('statresl_07_IN.pkl.gz')
    expfile = datadir('statresl_07_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_08():
    infile = datadir('statresl_08_IN.pkl.gz')
    expfile = datadir('statresl_08_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_09():
    infile = datadir('statresl_09_IN.pkl.gz')
    expfile = datadir('statresl_09_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_10():
    infile = datadir('statresl_10_IN.pkl.gz')
    expfile = datadir('statresl_10_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_11():
    infile = datadir('statresl_11_IN.pkl.gz')
    expfile = datadir('statresl_11_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_12():
    infile = datadir('statresl_12_IN.pkl.gz')
    expfile = datadir('statresl_12_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_13():
    infile = datadir('statresl_13_IN.pkl.gz')
    expfile = datadir('statresl_13_OUT.pkl.gz')
    dummy_test(infile, expfile)
