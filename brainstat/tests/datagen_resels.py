"""Data generation for compute_resels unit tests."""

import pickle

import numpy as np
from brainspace.mesh.mesh_elements import get_cells, get_points
from nilearn import datasets

from brainstat.context.utils import read_surface_gz
from brainstat.stats._multiple_comparisons import compute_resels
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect
from brainstat.tests.testutil import datadir


def generate_random_slm(rand_dict):
    """Generates a valid SLM for a surface.
    Parameters
    ----------
    surf : BSPolyData or a dictionary with key 'tri'
        Brain surface.
    Returns
    -------
    brainstat.stats.SLM
        SLM object.
    """
    # this is going to be the input slm
    I = {}
    rand_slm = SLM(FixedEffect(1), FixedEffect(1))
    for key in rand_dict.keys():
        setattr(rand_slm, key, rand_dict[key])
        I[key] = rand_dict[key]

    # this is going to be the output dict
    O = {}
    O["resels"], O["reselspvert"], O["edg"] = compute_resels(rand_slm)

    return I, O


def params2files(I, O, test_num):
    """Converts params to input/output files"""
    # filenames for the input and outpur dict's
    basename = "xstatresl"
    fin_name = datadir(basename + "_" + f"{test_num:02d}" + "_IN.pkl")
    fout_name = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")

    # save input and output data in pickle format with filenames above
    with open(fin_name, "wb") as f:
        pickle.dump(I, f, protocol=4)
    with open(fout_name, "wb") as g:
        pickle.dump(O, g, protocol=4)
    return


def generate_tests():
    # Test 01
    # ['tri'] will be a np array, shape (4, 3), int64
    np.random.seed(0)
    rand_dict = {}
    rand_dict["tri"] = np.random.randint(1, int(10), size=(4, 3))
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 1)

    # Test 02
    # ['tri'] :np array, shape (4, 3), int64
    # ['resl'] :np array, shape (8, 6), float64
    np.random.seed(0)
    rand_dict = {}
    n_vertices = 6
    rand_dict["tri"] = np.random.randint(1, n_vertices, size=(4, 3))
    rand_dict["resl"] = np.random.random_sample((8, n_vertices))
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 2)

    # Test 03
    # ['tri'] :np array, shape (4, 3), int64
    # ['resl'] :np array, shape (8, 6), float64
    # ['mask'] :np array, shape (5,), bool
    np.random.seed(0)
    rand_dict = {}
    n_vertices = 6
    rand_dict["tri"] = np.random.randint(1, n_vertices, size=(4, 3))
    rand_dict["resl"] = np.random.random_sample((8, n_vertices))
    rand_dict["mask"] = np.random.choice(
        a=[True, False], size=(rand_dict["tri"].max(),)
    )
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 3)

    # Test 04
    # ['lat'] :np array, shape (10, 10, 10), float64
    np.random.seed(0)
    rand_dict = {}
    rand_dict["lat"] = np.ones((10, 10, 10))
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 4)

    # Test 05
    # ['lat'] :np array, shape (10, 10, 10), bool
    np.random.seed(0)
    rand_dict = {}
    rand_dict["lat"] = np.random.choice(a=[False, True], size=(10, 10, 10))
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 5)

    # Test 06
    # ['tri] :np array, shape (1000,3)
    # ['mask'] :np array, shape (['tri'].max(),), bool
    np.random.seed(0)
    rand_dict = {}
    rand_dict["tri"] = np.random.randint(1, n_vertices, size=(1000, 3))
    rand_dict["mask"] = np.random.choice(
        a=[True, False], size=(rand_dict["tri"].max(),)
    )
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 6)

    # Test 07
    # ['lat'] :np array, shape (10, 10, 10), bool
    # ['resl'] :np array, shape (1359, 1), float64
    np.random.seed(0)
    rand_dict = {}
    rand_dict["lat"] = np.random.choice(a=[False, True], size=(10, 10, 10))
    rand_dict["resl"] = np.random.random_sample((1359, 1))
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 7)

    # Test 08
    # ['tri] :np array, shape (1000,3)
    # ['mask'] :np array, shape (499,), bool
    np.random.seed(1)
    rand_dict = {}
    rand_dict["tri"] = np.random.randint(1, 499, size=(1000, 3))
    rand_dict["mask"] = np.random.choice(
        a=[True, False], size=(rand_dict["tri"].max(),)
    )
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 8)

    # Test 09
    # ['lat'] :np array, shape (10, 10, 10), bool
    # ['resl'] :np array, shape (1198, 1), float64
    np.random.seed(1)
    rand_dict = {}
    rand_dict["lat"] = np.random.choice(a=[False, True], size=(10, 10, 10))
    rand_dict["resl"] = np.random.random_sample((1198, 1))
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 9)

    # Test 10
    # ['tri'] is pial_fs5, shape (20480, 3)
    pial_fs5 = datasets.fetch_surf_fsaverage()["pial_left"]
    pial_surf = read_surface_gz(pial_fs5)
    n_vertices = get_points(pial_surf).shape[0]
    rand_dict = {}
    rand_dict["tri"] = np.array(get_cells(pial_surf)) + 1
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 10)

    # Test 11
    # ['tri'] :pial_fs5, shape (20480, 3)
    # ['mask'] :np array, shape (['tri'].max(),), bool
    np.random.seed(0)
    rand_dict = {}
    rand_dict["tri"] = np.array(get_cells(pial_surf)) + 1
    rand_dict["mask"] = np.random.choice(
        a=[True, False], size=(rand_dict["tri"].max(),)
    )
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 11)

    # Test 12
    # ['tri'] :pial_fs5, shape (20480, 3) --> shuffle
    # ['mask'] :np array, shape (['tri'].max(),), bool
    np.random.seed(5)
    rand_dict = {}
    rand_dict["tri"] = np.array(get_cells(pial_surf)) + 1
    np.random.shuffle(rand_dict["tri"])
    rand_dict["mask"] = np.random.choice(
        a=[True, False], size=(rand_dict["tri"].max(),)
    )
    In, Out = generate_random_slm(rand_dict)
    params2files(In, Out, 12)


if __name__ == "__main__":
    generate_tests()
