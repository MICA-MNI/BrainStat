"""Data generation for peak_clus unit tests."""

import pickle

import numpy as np
from sklearn.model_selection import ParameterGrid

from brainstat.mesh.utils import mesh_edges
from brainstat.stats._multiple_comparisons import peak_clus
from brainstat.tests.testutil import datadir, generate_slm


def generate_random_slm(I):
    slm = generate_slm(
        t=I["t"],
        df=I["df"],
        k=I["k"],
        resl=I["resl"],
        tri=I["tri"],
        surf=I,
        mask=I["mask"],
        cluster_threshold=I["thresh"],
    )
    return slm


def generate_peak_clus_out(slm, I):
    D = {}
    D["peak"], D["clus"], D["clusid"] = peak_clus(
        slm, I["thresh"], reselspvert=I["reselspvert"], edg=I["edg"]
    )
    return D


def params2files(I, D, test_num):
    """Converts params to input/output files"""
    basename = "xstatpeakc"
    fin_name = datadir(basename + "_" + f"{test_num:02d}" + "_IN.pkl")
    fout_name = datadir(basename + "_" + f"{test_num:02d}" + "_OUT.pkl")
    with open(fin_name, "wb") as f:
        pickle.dump(I, f, protocol=4)
    with open(fout_name, "wb") as g:
        pickle.dump(D, g, protocol=4)
    return


def generate_test_data():
    np.random.seed(0)

    # generate the parameters
    tri = np.random.randint(1, int(50), size=(100, 3))
    coord = np.random.rand(3, 50)
    edg = mesh_edges({"tri": tri})
    n_edges = edg.shape[0]
    n_vertices = int(tri.shape[0])
    cluster_threshold = np.random.rand()
    mygrid = [
        {
            "num_t": [1, 2, 3],
            "k": [1, 2, 3],
            "df": [1, [1, 1]],
            "mask": [False, True],
            "reselspvert": [None, True],
        },
    ]
    myparamgrid = ParameterGrid(mygrid)

    # Generate data.
    test_num = 0
    for params in myparamgrid:
        I = {
            "tri": tri,
            "edg": edg,
            "thresh": cluster_threshold,
            "t": np.random.random_sample((params["num_t"], n_vertices)),
            "resl": np.random.random_sample((n_edges, 1)),
            "k": params["k"],
            "df": params["df"],
            "coord": coord,
        }

        if params["mask"] is True:
            I["mask"] = np.random.choice(a=[False, True], size=(n_vertices))
        else:
            I["mask"] = np.ones((n_vertices), dtype=bool)

        if params["reselspvert"] is True:
            I["reselspvert"] = np.random.rand(n_vertices)
        else:
            I["reselspvert"] = None

        # Here we go: generate slm & run peak_clus & save in-out
        slm = generate_random_slm(I)
        D = generate_peak_clus_out(slm, I)
        test_num += 1
        params2files(I, D, test_num)


if __name__ == "__main__":
    generate_test_data()
