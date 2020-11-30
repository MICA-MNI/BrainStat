import sys
sys.path.append("python")
from SurfStatEdg import *
import numpy as np
import pytest
import pickle


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    surf = {}

    if 'tri' in idic.keys():
        surf['tri'] = idic['tri']

    if 'lat' in idic.keys():
        surf['lat'] = idic['lat']

    # run SurfStatEdg
    out_edge = py_SurfStatEdg(surf)

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()
    exp_edge = expdic['edg']


    testout = []

    comp = np.allclose(out_edge, exp_edge, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01():
    infile  = './tests/data/unitdata/statedg_01_IN.pkl'
    expfile = './tests/data/unitdata/statedg_01_OUT.pkl'
    dummy_test(infile, expfile)


def test_02():
    infile  = './tests/data/unitdata/statedg_02_IN.pkl'
    expfile = './tests/data/unitdata/statedg_02_OUT.pkl'
    dummy_test(infile, expfile)


def test_03():
    infile  = './tests/data/unitdata/statedg_03_IN.pkl'
    expfile = './tests/data/unitdata/statedg_03_OUT.pkl'
    dummy_test(infile, expfile)


def test_04():
    infile  = './tests/data/unitdata/statedg_04_IN.pkl'
    expfile = './tests/data/unitdata/statedg_04_OUT.pkl'
    dummy_test(infile, expfile)


def test_05():
    infile  = './tests/data/unitdata/statedg_05_IN.pkl'
    expfile = './tests/data/unitdata/statedg_05_OUT.pkl'
    dummy_test(infile, expfile)


def test_06():
    infile  = './tests/data/unitdata/statedg_06_IN.pkl'
    expfile = './tests/data/unitdata/statedg_06_OUT.pkl'
    dummy_test(infile, expfile)



