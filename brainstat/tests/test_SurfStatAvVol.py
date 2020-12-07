from brainstat.stats import *
import surfstat_wrap as sw
import numpy as np
import random
import pytest
import os
import brainstat
sw.matlab_init_surfstat()

data_dir = (os.path.dirname(brainstat.__file__) +
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep + 'volfiles'
        + os.path.sep)


def dummy_test(filenames, fun = np.add, Nan = None, dimensionality = None):

    # wrap matlab functions
    M_data, M_vol = sw.matlab_AvVol(filenames, fun, Nan, dimensionality)

    # run python equivalent
    P_data, P_vol = SurfStatAvVol(filenames, fun, Nan)

    if filenames[0].endswith('.img'):
        P_data = P_data[:,:,:,0]

    # compare matlab-python outputs
    testout_AvVol = []

    for key in M_vol:
        testout_AvVol.append(np.allclose(np.squeeze(M_vol[key]),
                                                 np.squeeze(P_vol[key]),
                                                 rtol=1e-05, equal_nan=True))
    testout_AvVol.append(np.allclose(M_data, P_data,
                                 rtol=1e-05, equal_nan=True))

    assert all(flag == True for (flag) in testout_AvVol)


def test_01():
    # ANALYZE format (*img)
    filenames = [data_dir + 'Arandom' + str(x) + '.img' for x in range(1,6)]
    dummy_test(np.array(filenames))


def test_02():
    # ANALYZE format (*img)
    filenames = [data_dir + 'Arandom' + str(x) + '.img' for x in range(1,6)]
    dummy_test(np.array(filenames), fun=np.fmin)


def test_03():
    # ANALYZE format (*img)
    filenames = [data_dir + 'Arandom' + str(x) + '.img' for x in range(1,6)]
    dummy_test(np.array(filenames), fun=np.fmax)


def test_04():
    # ANALYZE format (*img), image with NaN values
    filenames = [data_dir + 'Arandom' + str(x) + '.img' for x in range(1,6)]
    filenames.append(data_dir + 'ArandomNaN.img')
    dummy_test(np.array(filenames), fun=np.add)


def test_05():
    # ANALYZE format (*img), image with NaN values, replace NaN
    filenames = [data_dir + 'Arandom' + str(x) + '.img' for x in range(1,6)]
    filenames.append(data_dir + 'ArandomNaN.img')
    dummy_test(np.array(filenames), fun=np.add, Nan = random.uniform(0, 50))


def test_06():
    # NIFTI files
    filenames = [data_dir + 'random' + str(x) + '.nii' for x in range(1,6)]
    dummy_test(np.array(filenames))


def test_07():
    # NIFTI files
    filenames = [data_dir + 'random' + str(x) + '.nii' for x in range(1,6)]
    dummy_test(np.array(filenames), fun=np.fmin)


def test_08():
    # NIFTI files
    filenames = [data_dir + 'random' + str(x) + '.nii' for x in range(1,6)]
    dummy_test(np.array(filenames), fun=np.fmax)


def test_09():
    # NIFTI files
    filenames = [data_dir + 'random' + str(x) + '.nii' for x in range(1,6)]
    filenames.append(data_dir + 'randomNaN.nii')
    dummy_test(np.array(filenames))


def test_10():
    # NIFTI files
    filenames = [data_dir + 'random' + str(x) + '.nii' for x in range(1,6)]
    filenames.append(data_dir + 'randomNaN.nii')
    dummy_test(np.array(filenames), fun=np.fmin)


def test_11():
    # NIFTI files
    filenames = [data_dir + 'random' + str(x) + '.nii' for x in range(1,6)]
    filenames.append(data_dir + 'randomNaN.nii')
    dummy_test(np.array(filenames), fun=np.add, Nan=3)
