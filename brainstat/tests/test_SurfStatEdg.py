import numpy as np
import pickle
import gzip
from .testutil import datadir
from ..stats import SurfStatEdg


def dummy_test(infile, expfile):

    # load input test data
    with gzip.open(infile, 'rb') as f:
        idic  = pickle.load(f)

    surf = {}

    if 'tri' in idic.keys():
        surf['tri'] = idic['tri']

    if 'lat' in idic.keys():
        surf['lat'] = idic['lat']

    # run SurfStatEdg
    out_edge = SurfStatEdg(surf)

    # load expected outout data
    with gzip.open(expfile, 'rb') as f:
        expdic  = pickle.load(f)
    exp_edge = expdic['edg']

    testout = []

    comp = np.allclose(out_edge, exp_edge, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = datadir('statedg_01_IN.pkl.gz')
    expfile = datadir('statedg_01_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_02():
    infile  = datadir('statedg_02_IN.pkl.gz')
    expfile = datadir('statedg_02_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_03():
    infile  = datadir('statedg_03_IN.pkl.gz')
    expfile = datadir('statedg_03_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_04():
    infile  = datadir('statedg_04_IN.pkl.gz')
    expfile = datadir('statedg_04_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_05():
    infile  = datadir('statedg_05_IN.pkl.gz')
    expfile = datadir('statedg_05_OUT.pkl.gz')
    dummy_test(infile, expfile)


def test_06():
    infile  = datadir('statedg_06_IN.pkl.gz')
    expfile = datadir('statedg_06_OUT.pkl.gz')
    dummy_test(infile, expfile)


