import testutil
import sys
sys.path.append("brainstat/stats")
from SurfStatEdg import SurfStatEdg
import numpy as np
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
    out_edge = SurfStatEdg(surf)

    # load expected outout data
    efile  = open(expfile, 'br')
    expdic = pickle.load(efile)
    efile.close()
    exp_edge = expdic['edg']


    testout = []

    comp = np.allclose(out_edge, exp_edge, rtol=1e-05, equal_nan=True)
    testout.append(comp)

    assert all(flag == True for (flag) in testout)


datadir = testutil.datadir


def test_01(datadir):
    infile  = datadir.join('statedg_01_IN.pkl')
    expfile = datadir.join('statedg_01_OUT.pkl')
    dummy_test(infile, expfile)


def test_02(datadir):
    infile  = datadir.join('statedg_02_IN.pkl')
    expfile = datadir.join('statedg_02_OUT.pkl')
    dummy_test(infile, expfile)


def test_03(datadir):
    infile  = datadir.join('statedg_03_IN.pkl')
    expfile = datadir.join('statedg_03_OUT.pkl')
    dummy_test(infile, expfile)


def test_04(datadir):
    infile  = datadir.join('statedg_04_IN.pkl')
    expfile = datadir.join('statedg_04_OUT.pkl')
    dummy_test(infile, expfile)


def test_05(datadir):
    infile  = datadir.join('statedg_05_IN.pkl')
    expfile = datadir.join('statedg_05_OUT.pkl')
    dummy_test(infile, expfile)


def test_06(datadir):
    infile  = datadir.join('statedg_06_IN.pkl')
    expfile = datadir.join('statedg_06_OUT.pkl')
    dummy_test(infile, expfile)


