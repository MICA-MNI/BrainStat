import json
import brainstat
from pathlib import Path


json_file = Path(brainstat.__file__).parent / "data_urls.json"


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
    }
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def read_data_fetcher_json() -> dict:
    """Reads the URLS of all external data from a .json file."""
    with open(json_file, "r") as f:
        return json.load(f)
