from sklearn.model_selection import ParameterGrid
import numpy as np
from brainstat.stats.terms import Term, Random
from brainstat.stats.SLM import SLM
from brainstat.context.utils import read_surface_gz
from nilearn.datasets import fetch_surf_fsaverage


def test_SLM():
    """Tests the SLM model using a grid of parameters

    Raises
    ------
    Exception
        First exception that occurs in computing the SLM.
    """
    samples = 10

    grid = list(create_parameter_grid(samples))
    Y = np.random.rand(samples, 10242, 3)
    for i in range(len(grid)):
        # Skip exceptions that we know error.
        if grid[i]["surf"] is None:
            if grid[i]["correction"] is not None and "rft" in grid[i]["correction"]:
                continue
        if grid[i]["Y_idx"] > 1 and grid[i]["two_tailed"] is False:
            continue

        try:
            slm = SLM(
                model=grid[i]["model"],
                contrast=grid[i]["contrast"],
                surf=grid[i]["surf"],
                mask=grid[i]["mask"],
                correction=grid[i]["correction"],
                two_tailed=grid[i]["two_tailed"],
            )
            slm.fit(Y[:, :, 0 : grid[i]["Y_idx"]])
        except Exception as e:
            print("Error on run:", i)
            print("SLM failed with the following parameters:")
            print("Model: ", grid[i]["model"])
            print("Contrast: ", grid[i]["contrast"])
            print("Surface: ", grid[i]["surf"])
            print("Mask: ", grid[i]["mask"])
            print("Correction: ", grid[i]["correction"])
            print("Two_tailed: ", grid[i]["two_tailed"])
            print("Y_idx: ", grid[i]["Y_idx"])
            raise e


def create_parameter_grid(samples):
    """Creates a parameter grid for the test function.

    Returns
    -------
    ParameterGrid
        All pairings of parameters to be run through the SLM class.
    """
    predictors = 3
    model = [
        Term(1) + Term(np.random.rand(samples, predictors), names=["y1", "y2", "y3"]),
    ]

    Y_idx = [1, 2, 3]
    contrast = [np.random.rand(samples), Term(np.random.rand(samples))]
    surf = [None, read_surface_gz(fetch_surf_fsaverage()["pial_left"])]
    mask = [None, np.random.rand(10242) > 0.1]
    correction = [None, ["rft", "fdr"]]
    two_tailed = [False, True]
    param_grid = ParameterGrid(
        {
            "Y_idx": Y_idx,
            "model": model,
            "contrast": contrast,
            "surf": surf,
            "mask": mask,
            "correction": correction,
            "two_tailed": two_tailed,
        }
    )
    return param_grid
