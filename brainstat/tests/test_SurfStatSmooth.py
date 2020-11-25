from brainstat.stats import *
import surfstat_wrap as sw
import numpy as np
import pytest
from scipy.io import loadmat
from nilearn.plotting.surf_plotting import load_surf_mesh
from nibabel.freesurfer.io import read_geometry
import os
import brainstat

sw.matlab_init_surfstat()


def dummy_test(Y, surf, FWHM):

    try:
        # wrap matlab functions
        Wrapped_Y = sw.matlab_Smooth(Y, surf, FWHM)
    except:
        pytest.skip("Original MATLAB code does not work with these inputs.")

    # run matlab equivalent
    Python_Y = Smooth(Y, surf, FWHM)

    # compare matlab-python outputs
    assert np.allclose(Wrapped_Y, Python_Y, rtol=1e-05, equal_nan=True)


def test_01():
    n = np.random.randint(1,100)
    Y = np.random.rand(n,n)
    surf = {}
    surf['tri'] = np.array([[1,2,3]])
    FWHM = 3.0
    dummy_test(Y, surf, FWHM)


def test_02():
    n = np.random.randint(1,100)
    Y = np.random.rand(n,n)
    m = np.random.randint(1,100)
    surf = {}
    surf['tri'] = np.random.randint(1,20,size=(m,3))
    FWHM = 3.0
    dummy_test(Y, surf, FWHM)


def test_03():
    n = np.random.randint(1,100)
    Y = np.random.rand(n,n)
    m = np.random.randint(1,100)
    surf = {}
    surf['lat'] = np.random.choice([0, 1], size=(3,3,3))
    FWHM = 3.0
    dummy_test(Y, surf, FWHM)


def test_04():
    n = np.random.randint(1,100)
    a = np.random.rand(n,3)
    b = np.random.rand(n,3)
    Y = np.zeros((n,3,2))
    Y[:,:,0] = a
    Y[:,:,1] = b
    surf = {}
    surf['tri'] = np.array([[1,2,3]])
    FWHM = 3.0
    dummy_test(Y, surf, FWHM)


def test_05():
    fname = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
        'surf_lsub.mat')
    f = loadmat(fname)
    surf = {}
    surf['tri'] = f['ave_lsub']['tri'][0,0]
    Y = np.random.rand(1, 1024)
    FWHM = 5.0
    dummy_test(Y, surf, FWHM)


def test_06():
    fname = (os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
        'surf_lsub.mat')
    f = loadmat(fname)
    surf = {}
    surf['tri'] = f['ave_lsub']['tri'][0,0]
    n = np.random.randint(1,100)
    Y = np.random.rand(n, 1024)
    FWHM = 5.0
    dummy_test(Y, surf, FWHM)


def test_07():
    # load freesurfer data lh.pial & rh.pial
    Pial_Mesh_Left  = load_surf_mesh(
        os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
        'lh.pial'
    )
    Pial_Mesh_Right = load_surf_mesh(
        os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
        'rh.pial'
    )
    Pial_Mesh = {}
    Pial_Mesh['tri'] = np.concatenate((Pial_Mesh_Left[1]+1,
                                       10242 + Pial_Mesh_Right[1]+1))
    Y = np.random.rand(1, 10242*2)
    FWHM = 2
    dummy_test(Y, Pial_Mesh, FWHM)


def test_08():
    # load freesurfer data lh.pial & rh.pial
    Pial_Mesh_Left  = load_surf_mesh(
        os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
        'lh.pial'
    )
    Pial_Mesh_Right = load_surf_mesh(
        os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
        'rh.pial'
    )
    Pial_Mesh = {}
    Pial_Mesh['tri'] = np.concatenate((Pial_Mesh_Left[1]+1,
                                       10242 + Pial_Mesh_Right[1]+1))
    n = np.random.randint(1,100)
    Y = np.random.rand(n, 10242*2)
    FWHM = 2
    dummy_test(Y, Pial_Mesh, FWHM)


def test_09():
    # load freesurfer data lh.white & rh.white
    SW_surf_L  = load_surf_mesh(
        os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
        'lh.white'
    )
    SW_surf_R = load_surf_mesh(
        os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
        'rh.white'
    )
    SW = {}
    SW['tri'] = np.concatenate((SW_surf_L[1]+1, 10242 + SW_surf_R[1]+1))
    Y = np.random.rand(1, 10242*2)
    FWHM = 2
    dummy_test(Y, SW, FWHM)


def test_10():
    # load freesurfer data lh.white & rh.white
    SW_surf_L  = load_surf_mesh(
        os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
        'lh.white'
    )
    SW_surf_R = load_surf_mesh(
        os.path.dirname(brainstat.__file__) + 
        os.path.sep + 'tests' + os.path.sep + 'data' + os.path.sep +
        'rh.white'
    )
    SW = {}
    SW['tri'] = np.concatenate((SW_surf_L[1]+1, 10242 + SW_surf_R[1]+1))
    n = np.random.randint(1,100)
    Y = np.random.rand(n, 10242*2)
    FWHM = np.random.rand() * 2
    dummy_test(Y, SW, FWHM)
