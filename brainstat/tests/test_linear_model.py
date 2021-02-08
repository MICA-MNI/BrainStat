import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats.models import linear_model
from brainstat.stats.terms import Term


def dummy_test(slm, oslm):

    testout = []

    for key in slm.keys():
        comp = np.allclose(slm[key], oslm[key], rtol=1e-05, equal_nan=True)
        testout.append(comp)

    print(testout)

    assert all(flag == True for (flag) in testout)


def test_01():
    # ['Y'] and ['M'] middle sized 2D arrays
    # ['Y'] : np array, shape (43, 43), dtype('float64')
    # ['M'] : np array, (43, 43), dtype('float64')
    infile = datadir("linmod_01_IN.pkl")
    expfile = datadir("linmod_01_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    M = idic["M"]
    slm = linear_model(Y, M)
    # oslm : expected output
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_02():
    # similar to test_01, shapes of ['Y'] and ['M'] changed
    # ['Y'] : np array, (62, 7), dtype('float64')
    # ['M'] : np array, (62, 92), dtype('float64')
    infile = datadir("linmod_02_IN.pkl")
    expfile = datadir("linmod_02_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    M = idic["M"]
    slm = linear_model(Y, M)
    # oslm : expected output
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_03():
    # ['Y'] is a 3D array, ['M'] is a 2D array
    # ['Y'] : np array, (52, 64, 76), dtype('float64')
    # ['M'] : np array, (52, 2), dtype('float64')
    infile = datadir("linmod_03_IN.pkl")
    expfile = datadir("linmod_03_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    M = idic["M"]
    slm = linear_model(Y, M)
    # oslm : expected output
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_04():
    # similar to test_03, shapes of ['Y'] and ['M'] changed
    # ['Y'] : np array, (69, 41, 5), dtype('float64')
    # ['M'] : np array, (69, 30), dtype('float64')
    infile = datadir("linmod_04_IN.pkl")
    expfile = datadir("linmod_04_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    M = idic["M"]
    slm = linear_model(Y, M)
    # oslm : expected output
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_05():
    # similar to test_01, shapes of ['Y'] and ['M'] changed
    # ['Y'] : np array, (81, 1), dtype('float64')
    # ['M'] : np array, (81, 2), dtype('float64')
    infile = datadir("linmod_05_IN.pkl")
    expfile = datadir("linmod_05_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    M = idic["M"]
    slm = linear_model(Y, M)
    # oslm : expected output
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_06():
    # similar to test_03, shapes of ['Y'] and ['M'] changed
    # ['Y'] : np array, (93, 41, 57), dtype('float64')
    # ['M'] : np array, (93, 67), dtype('float64')
    infile = datadir("linmod_06_IN.pkl")
    expfile = datadir("linmod_06_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    M = idic["M"]
    M = Term(M)
    slm = linear_model(Y, M)
    # oslm : expected output
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_07():
    # similar to test_03, shapes of ['Y'] and ['M'] changed
    # ['Y'] : np array, (40, 46, 21), dtype('float64')
    # ['M'] : np array, (40, 81), dtype('float64')
    infile = datadir("linmod_07_IN.pkl")
    expfile = datadir("linmod_07_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    M = idic["M"]
    M = Term(M)
    slm = linear_model(Y, M)
    # oslm : expected output
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_08():
    # ['Y'] and ['M'] mid. sized 2D arrays + optional ['tri'] input for surf
    # ['Y'] : np array, (93, 43), dtype('float64')
    # ['M'] : np array, (93, 2), dtype('float64')
    # ['tri'] : np array, (93, 3), dtype('int64')
    infile = datadir("linmod_08_IN.pkl")
    expfile = datadir("linmod_08_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    M = idic["M"]
    surf = {}
    surf["tri"] = idic["tri"]
    slm = linear_model(Y, M, surf)
    # oslm : expected output
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_09():
    # similar to test_03 + optional ['tri'] input for surf
    # ['Y'] : np array, (98, 69, 60), dtype('float64')
    # ['M'] : np array, (98, 91), dtype('float64')
    # ['tri'] : np array, (60, 3), dtype('int64')
    infile = datadir("linmod_09_IN.pkl")
    expfile = datadir("linmod_09_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    M = idic["M"]
    M = Term(M)
    surf = {}
    surf["tri"] = idic["tri"]
    slm = linear_model(Y, M, surf)
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_10():
    # similar to test_02 + optional ['lat'] input for surf
    # ['Y'] : np array, (49, 27), dtype('float64')
    # ['M'] : np array, (49, 2), dtype('float64')
    # ['lat'] : np array, (3, 3, 3), dtype('bool')
    infile = datadir("linmod_10_IN.pkl")
    expfile = datadir("linmod_10_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    M = idic["M"]
    surf = {}
    surf["lat"] = idic["lat"]
    slm = linear_model(Y, M, surf)
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_11():
    # similar to test_03 + optional ['lat'] input for surf
    # ['Y'] : np array, (45, 27, 3), dtype('float64')
    # ['M'] : np array, (45, 7), dtype('float64')
    # ['lat'] : np array, (3, 3, 3), dtype('int64')
    infile = datadir("linmod_11_IN.pkl")
    expfile = datadir("linmod_11_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    M = idic["M"]
    M = Term(M)
    surf = {}
    surf["lat"] = idic["lat"]
    slm = linear_model(Y, M, surf)
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_12():
    # real dataset, ['Y'] 20k columns, ['age'] modelling with Term, ['tri'] 40k vertex
    # ['Y'] : np array, (10, 20484), dtype('float64')
    # ['age'] : np array, (1, 10), dtype('float64')
    # ['tri'] : np array, (40960, 3), dtype('int32')
    # ['coord'] :np array, (3, 20484), dtype('float64')
    infile = datadir("linmod_12_IN.pkl")
    expfile = datadir("linmod_12_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    age = idic["age"]
    AGE = Term(np.array(age), "AGE")
    M = 1 + AGE
    surf = {}
    surf["tri"] = idic["tri"]
    surf["coord"] = idic["coord"]
    slm = linear_model(Y, M, surf)
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_13():
    # similar to test_12, ['Y'] values shuffled
    # ['Y'] : np array, (10, 20484), dtype('float64')
    # ['age'] : np array, (1, 10), dtype('float64')
    # ['tri'] : np array, (40960, 3), dtype('int32')
    # ['coord'] : np array, (3, 20484), dtype('float64')
    infile = datadir("linmod_13_IN.pkl")
    expfile = datadir("linmod_13_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    age = idic["age"]
    AGE = Term(np.array(age), "AGE")
    M = 1 + AGE
    surf = {}
    surf["tri"] = idic["tri"]
    surf["coord"] = idic["coord"]
    slm = linear_model(Y, M, surf)
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_14():
    # similar to test_12, ['Y'] and ['tri'] values shuffled
    # ['Y'] : np array, (10, 20484), dtype('float64')
    # ['age'] : np array, (1, 10), dtype('float64')
    # ['tri'] : np array, (40960, 3), dtype('int32')
    # ['coord'] : np array, (3, 20484), dtype('float64')
    infile = datadir("linmod_14_IN.pkl")
    expfile = datadir("linmod_14_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    age = idic["age"]
    AGE = Term(np.array(age), "AGE")
    M = 1 + AGE
    surf = {}
    surf["tri"] = idic["tri"]
    surf["coord"] = idic["coord"]
    slm = linear_model(Y, M, surf)
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_15():
    # choose ['Y']-values in range of [-1, 1], modeling from ['params'] & ['colnames']
    # ['Y'] : np array, (20, 20484), dtype('float64')
    # ['params'] : np array, (20, 9), dtype('uint16')
    # ['colnames'] : np array, (9,), dtype('<U11')
    # ['tri'] : np array, (40960, 3), dtype('int32')
    # ['coord'] : np array, (3, 20484), dtype('float64')
    infile = datadir("linmod_15_IN.pkl")
    expfile = datadir("linmod_15_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    params = idic["params"]
    colnames = list(idic["colnames"])
    M = Term(params, colnames)
    surf = {}
    surf["tri"] = idic["tri"]
    surf["coord"] = idic["coord"]
    slm = linear_model(Y, M, surf)
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)


def test_16():
    # similar to test_15, modeling only using the ['params']
    # ['Y'] : np array, (20, 20484), dtype('float64')
    # ['params'] : np array, (20, 9), dtype('uint16')
    # ['tri'] : np array, (40960, 3), dtype('int32')
    # ['coord'] :np array, (3, 20484), dtype('float64')
    infile = datadir("linmod_16_IN.pkl")
    expfile = datadir("linmod_16_OUT.pkl")
    ifile = open(infile, "br")
    idic = pickle.load(ifile)
    ifile.close()
    Y = idic["Y"]
    params = idic["params"]
    M = Term(params)
    surf = {}
    surf["tri"] = idic["tri"]
    surf["coord"] = idic["coord"]
    slm = linear_model(Y, M, surf)
    ofile = open(expfile, "br")
    oslm = pickle.load(ofile)
    ofile.close()
    dummy_test(slm, oslm)
