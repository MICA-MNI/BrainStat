"""Data generation for mesh_normalize unit tests."""

import pickle

import numpy as np
from brainspace.mesh.mesh_elements import get_cells
from nilearn import datasets
from sklearn.model_selection import ParameterGrid

from brainstat.context.utils import read_surface_gz
from brainstat.mesh.data import mesh_normalize
from brainstat.tests.testutil import datadir


def generate_mesh_normalize_out(I):
    if "mask" not in I.keys():
        I["mask"] = None
    # run mesh_normalize
    D = {}
    D["Python_Y"], D["Python_Yav"] = mesh_normalize(I["Y"], I["mask"], subdiv="s")
    return D


def params2files(I, D, test_num):
    """Converts params to input/output files"""
    basename = "xstatnor"
    fin_name = datadir(basename + "_" + f"{test_num:02d}" + "_IN.pkl")
    fout_name = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")
    with open(fin_name, "wb") as f:
        pickle.dump(I, f, protocol=4)
    with open(fout_name, "wb") as g:
        pickle.dump(D, g, protocol=4)
    return


def generate_test_data():
    pial_fs5 = datasets.fetch_surf_fsaverage()["pial_left"]
    pial_surf = read_surface_gz(pial_fs5)
    real_tri = np.array(get_cells(pial_surf))

    np.random.seed(0)

    mygrid = [
        {
            "Y": [
                np.random.randint(1, 10, size=(1, 20)),
                np.random.randint(1, 10, size=(2, 20)),
                np.random.randint(2, 10, size=(3, 20, 4)),
            ],
            "mask": [None, np.random.choice(a=[False, True], size=(20,))],
        },
        {
            "Y": [real_tri],
            "mask": [None, np.random.choice(a=[False, True], size=(real_tri.shape[1]))],
        },
    ]

    myparamgrid = ParameterGrid(mygrid)

    # Here wo go!
    # Tests 1-4 : Y is 2D or 3D arrays type int, mask is None or random bool
    # Tests 6-8 : Y is pial_fs5 trinagles, mask is None or random bool

    test_num = 0
    for params in myparamgrid:
        test_num += 1
        I = {}
        for key in params.keys():
            I[key] = params[key]
        D = generate_mesh_normalize_out(I)
        params2files(I, D, test_num)


if __name__ == "__main__":
    generate_test_data()
