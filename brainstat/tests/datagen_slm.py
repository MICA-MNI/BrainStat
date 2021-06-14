import pickle
import numpy as np
from sklearn.model_selection import ParameterGrid
import nibabel as nib
import os
import brainstat
from brainstat.tests.testutil import datadir, slm2dict
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect, MixedEffect
from brainstat.tutorial.utils import fetch_tutorial_data
from brainstat.context.utils import read_surface_gz
from brainspace.mesh.mesh_elements import get_cells, get_points
from nilearn.datasets import fetch_surf_fsaverage


def slm2files(slm, basename, test_num, input=True):
    """Converts an SLM to its input/output files

    Parameters
    ----------
    slm : brainstat.stats.SLM
        SLM object.
    basename : str
        Base name for the file.
    test_num : int
        Number of the test.
    """
    D = slm2dict(slm)
    D['model'] = 'fixed' if type(D['model']) is FixedEffect else 'mixed'
    if '_surf' in D and D['_surf'] is not None:
        D['surf'] = {'tri': get_cells(D['_surf']), 'coord': get_points(D['_surf'])}
        D.pop('_surf')
    
    if input:
        stage = "IN"
    else:
        stage = "OUT"
    filename = datadir(basename + "_" + f"{test_num:02d}" + "_" + stage + ".pkl")
    with open(filename, "wb") as f:
        pickle.dump(D, f, protocol=4)


def load_training_data(n):
    brainstat_dir = os.path.dirname(brainstat.__file__)
    data_dir = os.path.join(brainstat_dir, "tutorial")

    tutorial_data = fetch_tutorial_data(n_subjects=n, data_dir=data_dir)
    age = tutorial_data["demographics"]["AGE"].to_numpy()
    iq = tutorial_data["demographics"]["IQ"].to_numpy()

    # Reshape the thickness files such that left and right hemispheres are in the same row.
    files = np.reshape(np.array(tutorial_data["image_files"]), (-1, 2))

    thickness = np.zeros((n, 10242))
    for i in range(n):
        thickness[i, :] = np.squeeze(nib.load(files[i, 0]).get_fdata())
    mask = np.all(thickness != 0, axis=0)

    pial = read_surface_gz(fetch_surf_fsaverage()["pial_left"])
    return (pial, mask, age, iq, thickness)


def generate_test_data():
    pial, mask, age, iq, thickness = load_training_data(n=20)
    fixed_model = FixedEffect(1) + FixedEffect(age, "age")
    mixed_model = (
        FixedEffect(1)
        + FixedEffect(age, "age")
        + MixedEffect(iq, name_ran="iq")
        + MixedEffect(1, name_ran="Identity")
    )

    variates_2 = np.concatenate((thickness[:, :, None], np.random.random_sample(thickness.shape)[:,:,None]), axis=2)
    variates_3 = np.concatenate((thickness[:, :, None], np.random.rand(thickness.shape[0], thickness.shape[1], 2)), axis=2)

    # Separate RFT in the parameters as this requires a surface.
    parameters = [
        {
            "Y": [thickness, variates_2, variates_3],
            "model": [fixed_model, mixed_model],
            "contrast": [-age],
            "correction": [None, "fdr"],
            "surf": [None, pial],
            "mask": [None, mask],
            "niter": [1, 10],
            "thetalim": [0.01, 0.05],
            "drlim": [0.1, 0.2],
            "two_tailed": [True, False],
            "cluster_threshold": [0.001, 1.2],
        },
        {
            "Y": [thickness, variates_2, variates_3],
            "model": [fixed_model, mixed_model],
            "contrast": [-age],
            "surf": [pial],
            "mask": [None, mask],
            "correction": ["rft"],
            "niter": [1, 10],
            "thetalim": [0.01, 0.05],
            "drlim": [0.1, 0.2],
            "two_tailed": [True, False],
            "cluster_threshold": [0.001, 1.2],
        },
    ]

    test_num = 0
    for params in ParameterGrid(parameters):
        test_num += 1
        slm = SLM(
            params["model"],
            params["contrast"],
            params["surf"],
            params["mask"],
            correction=params["correction"],
            niter=params["niter"],
            thetalim=params["thetalim"],
            drlim=params["drlim"],
            two_tailed=params["two_tailed"],
            cluster_threshold=params["cluster_threshold"],
        )
        slm2files(slm, 'slm', test_num, input=True)
        print(type(params["model"]))
        try:
            slm.fit(params["Y"])
        except:
            import pdb
            pdb.set_trace()
        slm2files(slm, 'slm', test_num, input=False)


if __name__ == "__main__":
    generate_test_data()
