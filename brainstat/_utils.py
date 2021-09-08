"""Utilities for BrainStat developers."""
import json
import logging
import shutil
import urllib.request
import warnings
from pathlib import Path
from typing import Callable

import brainstat

json_file = Path(brainstat.__file__).parent / "data_urls.json"


BRAINSTAT_DATA_DIR = Path.home() / "brainstat_data"
data_directories = {
    "BRAINSTAT_DATA_DIR": BRAINSTAT_DATA_DIR,
    "ABIDE_DATA_DIR": BRAINSTAT_DATA_DIR / "abide_data",
    "BIGBRAIN_DATA_DIR": BRAINSTAT_DATA_DIR / "bigbrain_data",
    "MICS_DATA_DIR": BRAINSTAT_DATA_DIR / "mics_data",
    "NEUROSYNTH_DATA_DIR": BRAINSTAT_DATA_DIR / "neurosynth_data",
    "PARCELLATION_DATA_DIR": BRAINSTAT_DATA_DIR / "parcellation_data",
    "SURFACE_DATA_DIR": BRAINSTAT_DATA_DIR / "surface_data",
}


def generate_data_fetcher_json() -> None:
    """Stores the URLS of all external data in a .json file."""
    data = {
        "bigbrain_profiles": {
            "fsaverage": {
                "url": "https://box.bic.mni.mcgill.ca/s/znBp7Emls0mMW1a/download",
            },
            "fsaverage5": {
                "url": "https://box.bic.mni.mcgill.ca/s/N8zstvuRb4sNcSe/download",
            },
            "fslr32k": {
                "url": "https://box.bic.mni.mcgill.ca/s/6zKHcg9xXu5inPR/download",
            },
        },
        "neurosynth_precomputed": {
            "url": "https://box.bic.mni.mcgill.ca/s/GvislmLffbCIZoI/download",
            "n_files": 3228,
        },
        "parcellations": {
            "glasser": {
                "fsaverage": {
                    "url": (
                        "https://box.bic.mni.mcgill.ca/s/y2NMHXr47WOCtpp/download",
                        "https://box.bic.mni.mcgill.ca/s/Y0Fmd2tIF69Mqpt/download",
                    ),
                },
                "fsaverage5": {
                    "url": (
                        "https://box.bic.mni.mcgill.ca/s/Kg4VdWRt4NHvr3B/download",
                        "https://box.bic.mni.mcgill.ca/s/9sEXgVKi3VJ9pXV/download",
                    ),
                },
                "fslr32k": {
                    "url": (
                        "https://box.bic.mni.mcgill.ca/s/y2NMHXr47WOCtpp/download",
                        "https://box.bic.mni.mcgill.ca/s/Y0Fmd2tIF69Mqpt/download",
                    ),
                },
            },
        },
        "masks": {
            "civet41k": {
                "url": "https://box.bic.mni.mcgill.ca/s/9kzBetBCZkkqN6w/download"
            },
            "civet164k": {
                "url": "https://box.bic.mni.mcgill.ca/s/rei5HtTDvexlEPA/download"
            },
        },
        "abide_tutorial": {
            "summary_spreadsheet": {
                "url": "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
            },
        },
    }
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def read_data_fetcher_json() -> dict:
    """Reads the URLS of all external data from a .json file."""
    with open(json_file, "r") as f:
        return json.load(f)


def deprecated(message: str) -> Callable:
    """Decorator for deprecated functions.

    Parameters
    ----------
    message : str
        Message to return to the user.
    """

    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                "{} is a deprecated function and will be removed in a future version. {}".format(
                    func.__name__, message
                ),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator


def _download_file(url: str, output_file: Path, overwrite: bool = False) -> None:
    """Downloads a file.

    Parameters
    ----------
    url : str
        URL of the download.
    output_file : pathlib.Path
        Path object of the output file.
    overwrite : bool
        If true, overwrite existing files, defaults to False.
    """

    if output_file.exists() and not overwrite:
        logging.debug(str(output_file) + " already exists and will not be overwritten.")
        return

    logging.debug("Downloading " + str(output_file))
    with urllib.request.urlopen(url) as response, open(output_file, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


if __name__ == "__main__":
    generate_data_fetcher_json()
