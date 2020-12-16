from brainstat.stats.SurfStatResels import SurfStatResels
from brainstat.stats.SurfStatEdg import SurfStatEdg
import surfstat_wrap as sw
import numpy as np
import os
import brainstat
import pytest
from scipy.io import loadmat

sw.matlab_init_surfstat()


def dummy_test(slm, mask=None):

    # Run MATLAB
    try:
        mat_output = sw.matlab_Resels(slm, mask)
        # Deal with either 1 or 3 output arguments.
        # if not isinstance(mat_output, np.ndarray):
        #    mat_output = mat_output[0].tolist()
        #    mat_output[1] = np.squeeze(mat_output[1])
        # else:
        mat_output = mat_output.tolist()
        if isinstance(mat_output, float):
            mat_output = [mat_output]
    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # Run Python
    resels_py,  reselspvert_py,  edg_py = SurfStatResels(slm, mask)
    if len(mat_output) == 1:
        py_output = [resels_py]
    else:
        py_output = [resels_py,
                     reselspvert_py,
                     edg_py+1]

    # compare matlab-python outputs
    test_out = []
    for py, mat in zip(py_output, mat_output):
        result = np.allclose(np.squeeze(py),
                             np.squeeze(np.asarray(mat)),
                             rtol=1e-05, equal_nan=True)
        test_out.append(result)

    assert all(flag == True for (flag) in test_out)


def test_01():
    # Test with only slm.tri
    slm = {'tri': np.array(
        [[1, 2, 3],
         [2, 3, 4],
         [1, 2, 4],
         [2, 3, 5]])}
    dummy_test(slm)


def test_02():
    # Test with slm.tri and slm.resl
    slm = {'tri': np.array(
        [[1, 2, 3],
         [2, 3, 4],
         [1, 2, 4],
         [2, 3, 5]]),
        'resl': np.random.rand(8, 6)}
    dummy_test(slm)


def test_03():
    # Test with slm.tri, slm.resl, and mask
    slm = {'tri': np.array(
        [[1, 2, 3],
         [2, 3, 4],
         [1, 2, 4],
         [2, 3, 5]]),
        'resl': np.random.rand(8, 6)}
    mask = np.array([True, True, True, False, True])
    dummy_test(slm, mask)


def test_04():
    # Test with slm.lat, 1's only.
    slm = {'lat': np.ones((10, 10, 10))}
    dummy_test(slm)


def test_05():
    # Test with slm.lat, both 0's and 1's.
    slm = {'lat': np.random.rand(10, 10, 10) > 0.5}
    dummy_test(slm)


def test_06():
    # Test with slm.lat, both 0's and 1's, and a mask.
    slm = {'lat': np.random.rand(10, 10, 10) > 0.5}
    mask = np.random.choice([False, True], np.sum(slm['lat']))
    dummy_test(slm, mask)


def test_07():
    # Test with slm.lat and slm.resl
    slm = {'lat': np.random.rand(10, 10, 10) > 0.5}
    edg = SurfStatEdg(slm)
    slm['resl'] = np.random.rand(edg.shape[0], 1)
    dummy_test(slm)


def test_08():
    # Test with slm.lat, slm.resl, and a mask
    slm = {'lat': np.random.rand(10, 10, 10) > 0.5}
    mask = np.random.choice([False, True], np.sum(slm['lat']))
    edg = SurfStatEdg(slm)
    slm['resl'] = np.random.rand(edg.shape[0], 1)
    dummy_test(slm, mask)


def test_09():
    # Test with slm.lat, slm.resl, and a fully false mask
    slm = {'lat': np.random.rand(10, 10, 10) > 0.5}
    mask = np.zeros(np.sum(slm['lat']), dtype=bool)
    edg = SurfStatEdg(slm)
    slm['resl'] = np.random.rand(edg.shape[0], 1)
    dummy_test(slm, mask)


def test_10():
    slmfile = (os.path.dirname(brainstat.__file__) +
               os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
               'slm.mat')
    slmdata = loadmat(slmfile)
    slm = {}
    slm['tri'] = slmdata['slm']['tri'][0, 0]
    slm['resl'] = slmdata['slm']['resl'][0, 0]
    dummy_test(slm)


def test_11():
    # real data & random mask
    slmfile = (os.path.dirname(brainstat.__file__) +
               os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
               'slm.mat')
    slmdata = loadmat(slmfile)
    slm = {}
    slm['tri'] = slmdata['slm']['tri'][0, 0]
    slm['resl'] = slmdata['slm']['resl'][0, 0]
    # v is number of vertices
    v = slm['tri'].max()
    mask = np.random.choice([False, True], v)
    dummy_test(slm, mask)


def test_12():
    # randomized (shuffled) real data
    slmfile = (os.path.dirname(brainstat.__file__) +
               os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
               'slm.mat')
    slmdata = loadmat(slmfile)
    slm = {}
    slm['tri'] = slmdata['slm']['tri'][0, 0]
    slm['resl'] = slmdata['slm']['resl'][0, 0]
    np.random.shuffle(slm['tri'])
    np.random.shuffle(slm['resl'])
    dummy_test(slm)


def test_13():
    # randomized (shuffled) real data & random mask
    slmfile = (os.path.dirname(brainstat.__file__) +
               os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
               'slm.mat')
    slmdata = loadmat(slmfile)
    slm = {}
    slm['tri'] = slmdata['slm']['tri'][0, 0]
    slm['resl'] = slmdata['slm']['resl'][0, 0]
    np.random.shuffle(slm['tri'])
    np.random.shuffle(slm['resl'])
    # v is number of vertices
    v = slm['tri'].max()
    mask = np.random.choice([False, True], v)
    dummy_test(slm, mask)
