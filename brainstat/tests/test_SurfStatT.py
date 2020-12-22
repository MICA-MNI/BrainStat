import testutil
import sys
sys.path.append("brainstat/stats")
from SurfStatT import SurfStatT
import numpy as np
import pickle


def dummy_test(infile, expfile):

    # load input test data
    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

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
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()

    testout = []

    for key in outdic.keys():
        comp = np.allclose(outdic[key], expdic[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    assert all(flag == True for (flag) in testout)

datadir = testutil.datadir


def test_01(datadir):
    infile  = datadir.join('statt_01_IN.pkl')
    expfile = datadir.join('statt_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02(datadir):
    infile  = datadir.join('statt_02_IN.pkl')
    expfile = datadir.join('statt_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03(datadir):
    infile  = datadir.join('statt_03_IN.pkl')
    expfile = datadir.join('statt_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04(datadir):
    infile  = datadir.join('statt_04_IN.pkl')
    expfile = datadir.join('statt_04_OUT.pkl')
    dummy_test(infile, expfile)


def test_05(datadir):
    infile  = datadir.join('statt_05_IN.pkl')
    expfile = datadir.join('statt_05_OUT.pkl')
    dummy_test(infile, expfile)


def test_06(datadir):
    infile  = datadir.join('statt_06_IN.pkl')
    expfile = datadir.join('statt_06_OUT.pkl')
    dummy_test(infile, expfile)


def test_07(datadir):
    infile  = datadir.join('statt_07_IN.pkl')
    expfile = datadir.join('statt_07_OUT.pkl')
    dummy_test(infile, expfile)


def test_08(datadir):
    infile  = datadir.join('statt_08_IN.pkl')
    expfile = datadir.join('statt_08_OUT.pkl')
    dummy_test(infile, expfile)


def test_09(datadir):
    infile  = datadir.join('statt_09_IN.pkl')
    expfile = datadir.join('statt_09_OUT.pkl')
    dummy_test(infile, expfile)


