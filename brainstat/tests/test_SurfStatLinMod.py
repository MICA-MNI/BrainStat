import numpy as np
import pickle
from .testutil import datadir
from ..stats import SurfStatLinMod
from ..stats import Term


def dummy_test(slm, oslm):

    testout = []

    for key in slm.keys():
        comp = np.allclose(slm[key], oslm[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    print(testout)

    assert all(flag == True for (flag) in testout)


def test_01(datadir):

    infile  = datadir.join('linmod_01_IN.pkl')
    expfile = datadir.join('linmod_01_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_02(datadir):

    infile  = datadir.join('linmod_02_IN.pkl')
    expfile = datadir.join('linmod_02_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_03(datadir):

    infile  = datadir.join('linmod_03_IN.pkl')
    expfile = datadir.join('linmod_03_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_04(datadir):

    infile  = datadir.join('linmod_04_IN.pkl')
    expfile = datadir.join('linmod_04_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_05(datadir):

    infile  = datadir.join('linmod_05_IN.pkl')
    expfile = datadir.join('linmod_05_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    slm = SurfStatLinMod(Y, M)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_06(datadir):

    infile  = datadir.join('linmod_06_IN.pkl')
    expfile = datadir.join('linmod_06_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    M   = Term(M)
    slm = SurfStatLinMod(Y, M)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_07(datadir):

    infile  = datadir.join('linmod_07_IN.pkl')
    expfile = datadir.join('linmod_07_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    M   = Term(M)
    slm = SurfStatLinMod(Y, M)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_08(datadir):

    infile  = datadir.join('linmod_08_IN.pkl')
    expfile = datadir.join('linmod_08_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    M    = idic['M']
    surf = {}
    surf['tri'] = idic['tri']
    slm = SurfStatLinMod(Y, M, surf)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_09(datadir):

    infile  = datadir.join('linmod_09_IN.pkl')
    expfile = datadir.join('linmod_09_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    M    = idic['M']
    M    = Term(M)
    surf = {}
    surf['tri'] = idic['tri']
    slm = SurfStatLinMod(Y, M, surf)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_10(datadir):

    infile  = datadir.join('linmod_10_IN.pkl')
    expfile = datadir.join('linmod_10_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    M    = idic['M']
    surf = {}
    surf['lat'] = idic['lat']
    slm = SurfStatLinMod(Y, M, surf)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_11(datadir):

    infile  = datadir.join('linmod_11_IN.pkl')
    expfile = datadir.join('linmod_11_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    M    = idic['M']
    M    = Term(M)
    surf = {}
    surf['lat'] = idic['lat']
    slm = SurfStatLinMod(Y, M, surf)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_12(datadir):

    infile  = datadir.join('linmod_12_IN.pkl')
    expfile = datadir.join('linmod_12_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_13(datadir):

    infile  = datadir.join('linmod_13_IN.pkl')
    expfile = datadir.join('linmod_13_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_14(datadir):

    infile  = datadir.join('linmod_14_IN.pkl')
    expfile = datadir.join('linmod_14_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_15(datadir):

    infile  = datadir.join('linmod_15_IN.pkl')
    expfile = datadir.join('linmod_15_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y        = idic['Y']
    params   = idic['params']
    colnames = list(idic['colnames'])
    M        = Term(params, colnames)

    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_16(datadir):

    infile  = datadir.join('linmod_16_IN.pkl')
    expfile = datadir.join('linmod_16_OUT.pkl')

    ifile = open(infile, 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y        = idic['Y']
    params   = idic['params']
    M        = Term(params)

    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = SurfStatLinMod(Y, M, surf)

    ofile = open(expfile, 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


