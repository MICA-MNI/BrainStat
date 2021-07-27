"""Data generation for mesh_edges unit tests."""

import pickle

import numpy as np

from brainstat.mesh.utils import mesh_edges
from brainstat.tests.testutil import datadir


def generate_random_mesh_edge_data(
    key_dim,
    finname,
    key_name="tri",
    key_dtype="float",
    seed=0,
):
    """Generate random test datasets."""
    # key_dim : tuple, dimensions of array
    # finname : string, filename ending with *pkl
    # key_name : string, "tri" or "lat"
    # key_dtype : string, float or int

    np.random.seed(seed=seed)

    surf = {}

    if key_name == "tri":
        surf["tri"] = np.random.randint(1, 1024, size=key_dim)
    elif key_name == "lat":
        surf["lat"] = np.random.randint(0, 2, size=key_dim)
    surf[key_name] = surf[key_name].astype(key_dtype)

    with open(finname, "wb") as handle:
        pickle.dump(surf, handle, protocol=4)
    return surf


def get_meshedge_output(surf, foutname):
    """Runs mesh_edges and returns all relevant output."""

    # run mesh_edges
    surf_out = {}
    surf_out["edg"] = mesh_edges(surf)

    with open(foutname, "wb") as handle:
        pickle.dump(surf_out, handle, protocol=4)  #

    return


def generate_data_test_mesh_edges():

    ### test_01 data in-out generation
    print("test_mesh_edges.py : test_01 data is generated..")
    # ['tri'] is a 2D numpy array of shape (78, 3), dtype('float64')
    key_dim = (78, 3)
    finname = datadir("xstatedg_01_IN.pkl")
    key_name = "tri"
    key_dtype = np.float64
    seed = 444
    surf = generate_random_mesh_edge_data(key_dim, finname, key_name, key_dtype, seed)
    foutname = datadir("xstatedg_01_OUT.pkl")
    get_meshedge_output(surf, foutname)

    ### test_02 data in-out generation
    print("test_mesh_edges.py : test_02 data is generated..")
    # ['lat'] is a 2D numpy array of shape (10, 10), dtype('float64')
    key_dim = (10, 10)
    finname = datadir("xstatedg_02_IN.pkl")
    key_name = "lat"
    key_dtype = np.float64
    seed = 445
    surf = generate_random_mesh_edge_data(key_dim, finname, key_name, key_dtype, seed)
    foutname = datadir("xstatedg_02_OUT.pkl")
    get_meshedge_output(surf, foutname)

    ### test_03 data in-out generation
    print("test_mesh_edges.py : test_03 data is generated..")
    # ['lat'] is a 3D numpy array of shape (10, 10, 10), dtype('int64')
    key_dim = (10, 10, 10)
    finname = datadir("xstatedg_03_IN.pkl")
    key_name = "lat"
    key_dtype = np.int64
    seed = 446
    surf = generate_random_mesh_edge_data(key_dim, finname, key_name, key_dtype, seed)
    foutname = datadir("xstatedg_03_OUT.pkl")
    get_meshedge_output(surf, foutname)

    ### test_04 data in-out generation
    print("test_mesh_edges.py : test_04 data is generated..")
    # ['tri'] is a 2D numpy array of shape (2044, 3), dtype('uint16')
    key_dim = (2044, 3)
    finname = datadir("xstatedg_04_IN.pkl")
    key_name = "tri"
    key_dtype = np.uint16
    seed = 447
    surf = generate_random_mesh_edge_data(key_dim, finname, key_name, key_dtype, seed)
    foutname = datadir("xstatedg_04_OUT.pkl")
    get_meshedge_output(surf, foutname)


if __name__ == "__main__":
    generate_data_test_mesh_edges()
