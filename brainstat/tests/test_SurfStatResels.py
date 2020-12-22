import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatResels


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

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

    resels_py, reselspvert_py, edg_py =  SurfStatResels(slm,mask)

    out = {}
    out['resels']      = resels_py
    out['reselspvert'] = reselspvert_py
    out['edg']         = edg_py

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in out.keys():
        if out[key] is not None and expdic[key] is not None:
            comp = np.allclose(out[key], expdic[key], rtol=1e-05, equal_nan=True)
            testout.append(comp)

    assert all(flag == True for (flag) in testout)


def test_01(datadir):
    infile  = datadir.join('statresl_01_IN.pkl')
    expfile = datadir.join('statresl_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02(datadir):
    infile  = datadir.join('statresl_02_IN.pkl')
    expfile = datadir.join('statresl_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03(datadir):
    infile  = datadir.join('statresl_03_IN.pkl')
    expfile = datadir.join('statresl_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04(datadir):
    infile  = datadir.join('statresl_04_IN.pkl')
    expfile = datadir.join('statresl_04_OUT.pkl')
    dummy_test(infile, expfile)


def test_05(datadir):
    infile  = datadir.join('statresl_05_IN.pkl')
    expfile = datadir.join('statresl_05_OUT.pkl')
    dummy_test(infile, expfile)


def test_06(datadir):
    infile  = datadir.join('statresl_06_IN.pkl')
    expfile = datadir.join('statresl_06_OUT.pkl')
    dummy_test(infile, expfile)


def test_07(datadir):
    infile  = datadir.join('statresl_07_IN.pkl')
    expfile = datadir.join('statresl_07_OUT.pkl')
    dummy_test(infile, expfile)


def test_08(datadir):
    infile  = datadir.join('statresl_08_IN.pkl')
    expfile = datadir.join('statresl_08_OUT.pkl')
    dummy_test(infile, expfile)


def test_09(datadir):
    infile  = datadir.join('statresl_09_IN.pkl')
    expfile = datadir.join('statresl_09_OUT.pkl')
    dummy_test(infile, expfile)


def test_10(datadir):
    infile  = datadir.join('statresl_10_IN.pkl')
    expfile = datadir.join('statresl_10_OUT.pkl')
    dummy_test(infile, expfile)


def test_11(datadir):
    infile  = datadir.join('statresl_11_IN.pkl')
    expfile = datadir.join('statresl_11_OUT.pkl')
    dummy_test(infile, expfile)


def test_12(datadir):
    infile  = datadir.join('statresl_12_IN.pkl')
    expfile = datadir.join('statresl_12_OUT.pkl')
    dummy_test(infile, expfile)


def test_13(datadir):
    infile  = datadir.join('statresl_13_IN.pkl')
    expfile = datadir.join('statresl_13_OUT.pkl')
    dummy_test(infile, expfile)


