"""Data generation for stat_threshold unit tests."""

import pickle

import numpy as np
from sklearn.model_selection import ParameterGrid

from brainstat.stats._multiple_comparisons import stat_threshold
from brainstat.tests.testutil import datadir


def generate_stat_threshold_out(I):
    D = {}
    (
        D["peak_threshold"],
        D["extent_threshold"],
        D["peak_threshold_1"],
        D["extent_threshold_1"],
        D["t"],
        D["rho"],
    ) = stat_threshold(
        I["search_volume"],
        I["num_voxels"],
        I["fwhm"],
        I["df"],
        I["p_val_peak"],
        I["cluster_threshold"],
        I["p_val_extent"],
        I["nconj"],
        I["nvar"],
        None,
        None,
        I["nprint"],
    )
    return D


def params2files(I, D, test_num):
    """Converts params to input/output files"""
    basename = "xthresh"
    fin_name = datadir(basename + "_" + f"{test_num:02d}" + "_IN.pkl")
    fout_name = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")
    with open(fin_name, "wb") as f:
        pickle.dump(I, f, protocol=4)
    with open(fout_name, "wb") as g:
        pickle.dump(D, g, protocol=4)
    return


def generate_test_data():
    np.random.seed(0)

    mygrid = [
        {
            "search_volume": [
                np.random.rand() * 5,
                np.random.randint(1, 20, size=(3,)).tolist(),
                np.random.rand(2, 2) * 20,
            ],
            "num_voxels": [
                int(1),
                np.random.randint(1, 10, size=(3,)).tolist(),
            ],
            "fwhm": [0.0, np.random.rand() * 5],
            "df": [5, np.random.randint(10, size=(2, 2))],
            "p_val_peak": [
                0.05,
                np.random.rand(
                    4,
                ).tolist(),
            ],
            "cluster_threshold": [0.001],
            "p_val_extent": [0.05],
            "nconj": [0.5],
            "nvar": [1],
            "nprint": [0],
        }
    ]

    myparamgrid = ParameterGrid(mygrid)

    # Here wo go!
    # Tests 1-48 : search_volume -> float, list, 2D array
    # num_voxel -> int, list of ints,
    # fwhm -> 0, float, df -> int, 2D array of type ints,
    # p_val_peak --> 0.05, list of floats
    # cluster_threshold, p_val_extent, nconj, nvar, nprint -> default values

    test_num = 0
    for params in myparamgrid:
        test_num += 1
        I = {}
        for key in params.keys():
            I[key] = params[key]
        D = generate_stat_threshold_out(I)
        params2files(I, D, test_num)


if __name__ == "__main__":
    generate_test_data()
