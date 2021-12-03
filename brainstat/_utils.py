"""Utilities for BrainStat developers."""
import json
import logging
import logging.config
import shutil
import urllib.request
import warnings
from pathlib import Path
from typing import Callable

import brainstat

json_file = Path(brainstat.__file__).parent / "data" / "data_urls.json"

logging.config.fileConfig(Path(brainstat.__file__).parent / "data" / "logging.conf")
logger = logging.getLogger("brainstat")


BRAINSTAT_DATA_DIR = Path.home() / "brainstat_data"
data_directories = {
    "BRAINSTAT_DATA_DIR": BRAINSTAT_DATA_DIR,
    "ABIDE_DATA_DIR": BRAINSTAT_DATA_DIR / "abide_data",
    "BIGBRAIN_DATA_DIR": BRAINSTAT_DATA_DIR / "bigbrain_data",
    "GRADIENT_DATA_DIR": BRAINSTAT_DATA_DIR / "gradient_data",
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
        "gradients": {
            "margulies2016": {
                "url": "https://box.bic.mni.mcgill.ca/s/LWFaQlOxUWmRlc0/download",
            }
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
            "yeo": {"url": "https://box.bic.mni.mcgill.ca/s/vcSXEk1wx0jN86N/download"},
        },
        "masks": {
            "civet41k": {
                "url": "https://box.bic.mni.mcgill.ca/s/9kzBetBCZkkqN6w/download"
            },
            "civet164k": {
                "url": "https://box.bic.mni.mcgill.ca/s/rei5HtTDvexlEPA/download"
            },
            "fsaverage5": {
                "url": "https://box.bic.mni.mcgill.ca/s/jsSDyiDcm9KEQpf/download"
            },
            "fsaverage": {
                "url": "https://box.bic.mni.mcgill.ca/s/XiZx9HfHaufb4kD/download"
            },
            "fslr32k": {
                "url": "https://box.bic.mni.mcgill.ca/s/cFXCrSkfiJFjUJ0/download"
            },
        },
        "spheres": {
            "civet41k": {
                "url": "https://box.bic.mni.mcgill.ca/s/9fXWyAd7gLJu7C8/download",
            },
        },
        "abide_tutorial": {
            "summary_spreadsheet": {
                "url": "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
            },
        },
        "mics_tutorial": {
            "demographics": {
                "url": "https://box.bic.mni.mcgill.ca/s/7bW9JIpvQvSJeuf/download"
            },
            "thickness": {
                "url": "https://box.bic.mni.mcgill.ca/s/kMi6lU91piwdaCf/download"
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


def _download_file(
    url: str, output_file: Path, overwrite: bool = False, verbose=True
) -> None:
    """Downloads a file.

    Parameters
    ----------
    url : str
        URL of the download.
    output_file : pathlib.Path
        Path object of the output file.
    overwrite : bool
        If true, overwrite existing files, defaults to False.
    verbose : bool
        If true, print a download message, defaults to True.
    """

    if output_file.exists() and not overwrite:
        return

    if verbose:
        logger.info("Downloading " + str(output_file) + " from " + url + ".")
    with urllib.request.urlopen(url) as response, open(output_file, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


if __name__ == "__main__":
    generate_data_fetcher_json()
