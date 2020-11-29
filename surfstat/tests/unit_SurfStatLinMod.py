import sys
sys.path.append("python")
import numpy as np
import pytest
from SurfStatLinMod import py_SurfStatLinMod
from term import Term
import pickle


def dummy_test(slm, oslm):

    testout = []

    for key in slm.keys():
        comp = np.allclose(slm[key], oslm[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    print(testout)

    assert all(flag == True for (flag) in testout)


def test_01():

    ifile = open('./tests/data/unitdata/linmod_01_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    slm = py_SurfStatLinMod(Y, M)

    ofile = open('./tests/data/unitdata/linmod_01_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_02():

    ifile = open('./tests/data/unitdata/linmod_02_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    slm = py_SurfStatLinMod(Y, M)

    ofile = open('./tests/data/unitdata/linmod_02_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_03():

    ifile = open('./tests/data/unitdata/linmod_03_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    slm = py_SurfStatLinMod(Y, M)

    ofile = open('./tests/data/unitdata/linmod_03_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_04():

    ifile = open('./tests/data/unitdata/linmod_04_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    slm = py_SurfStatLinMod(Y, M)

    ofile = open('./tests/data/unitdata/linmod_04_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_05():

    ifile = open('./tests/data/unitdata/linmod_05_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    slm = py_SurfStatLinMod(Y, M)

    ofile = open('./tests/data/unitdata/linmod_05_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_06():

    ifile = open('./tests/data/unitdata/linmod_06_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    M   = Term(M)
    slm = py_SurfStatLinMod(Y, M)

    ofile = open('./tests/data/unitdata/linmod_06_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_07():

    ifile = open('./tests/data/unitdata/linmod_07_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y   = idic['Y']
    M   = idic['M']
    M   = Term(M)
    slm = py_SurfStatLinMod(Y, M)

    ofile = open('./tests/data/unitdata/linmod_07_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_08():

    ifile = open('./tests/data/unitdata/linmod_08_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    M    = idic['M']
    surf = {}
    surf['tri'] = idic['tri']
    slm = py_SurfStatLinMod(Y, M, surf)

    ofile = open('./tests/data/unitdata/linmod_08_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_09():

    ifile = open('./tests/data/unitdata/linmod_09_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    M    = idic['M']
    M    = Term(M)
    surf = {}
    surf['tri'] = idic['tri']
    slm = py_SurfStatLinMod(Y, M, surf)

    ofile = open('./tests/data/unitdata/linmod_09_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_10():

    ifile = open('./tests/data/unitdata/linmod_10_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    M    = idic['M']
    surf = {}
    surf['lat'] = idic['lat']
    slm = py_SurfStatLinMod(Y, M, surf)

    ofile = open('./tests/data/unitdata/linmod_10_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_11():

    ifile = open('./tests/data/unitdata/linmod_11_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    M    = idic['M']
    M    = Term(M)
    surf = {}
    surf['lat'] = idic['lat']
    slm = py_SurfStatLinMod(Y, M, surf)

    ofile = open('./tests/data/unitdata/linmod_11_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_12():

    ifile = open('./tests/data/unitdata/linmod_12_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = py_SurfStatLinMod(Y, M, surf)

    ofile = open('./tests/data/unitdata/linmod_12_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_13():

    ifile = open('./tests/data/unitdata/linmod_13_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = py_SurfStatLinMod(Y, M, surf)

    ofile = open('./tests/data/unitdata/linmod_13_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_14():

    ifile = open('./tests/data/unitdata/linmod_14_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y    = idic['Y']
    age  = idic['age']
    AGE  = Term(np.array(age), 'AGE')
    M    = 1 + AGE
    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = py_SurfStatLinMod(Y, M, surf)

    ofile = open('./tests/data/unitdata/linmod_14_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_15():

    ifile = open('./tests/data/unitdata/linmod_15_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y        = idic['Y']
    params   = idic['params']
    colnames = list(idic['colnames'])
    M        = Term(params, colnames)

    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = py_SurfStatLinMod(Y, M, surf)

    ofile = open('./tests/data/unitdata/linmod_15_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)


def test_16():

    ifile = open('./tests/data/unitdata/linmod_16_IN.pkl', 'br')
    idic  = pickle.load(ifile)
    ifile.close()

    Y        = idic['Y']
    params   = idic['params']
    M        = Term(params)

    surf = {}
    surf['tri'] = idic['tri']
    surf['coord'] = idic['coord']
    slm = py_SurfStatLinMod(Y, M, surf)

    ofile = open('./tests/data/unitdata/linmod_16_OUT.pkl', 'br')
    oslm  = pickle.load(ofile)
    ofile.close()

    dummy_test(slm, oslm)
