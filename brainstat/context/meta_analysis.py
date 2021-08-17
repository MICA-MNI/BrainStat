""" Meta-analytic decoding based on NiMARE """
import logging
import re
import tempfile
import urllib
import zipfile
from pathlib import Path
from typing import Generator, Optional, Sequence, Union

import nibabel as nib
import numpy as np
import pandas as pd
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from nilearn.datasets import load_mni152_brain_mask
from scipy.stats.stats import pearsonr

from brainstat._utils import data_directories, read_data_fetcher_json
from brainstat.mesh.interpolate import multi_surface_to_volume


def surface_decoder(
    pial: Union[str, BSPolyData, Sequence[Union[str, BSPolyData]]],
    white: Union[str, BSPolyData, Sequence[Union[str, BSPolyData]]],
    stat_labels: Union[str, np.ndarray, Sequence[Union[str, np.ndarray]]],
    *,
    interpolation: str = "linear",
    data_dir: Optional[Union[str, Path]] = None,
    database: str = "neurosynth",
) -> pd.DataFrame:
    """Meta-analytic decoding of surface maps using NeuroSynth or NeuroQuery.

    Parameters
    ----------
    pial : str, BSPolyData, sequence of str or BSPolyData
        Path of a pial surface file, BSPolyData of a pial surface or a list
        containing multiple of the aforementioned.
    white : str, BSPolyData, sequence of str or BSPolyData
        Path of a white matter surface file, BSPolyData of a pial surface or a
        list containing multiple of the aforementioned.
    stat_labels : str, numpy.ndarray, sequence of str or numpy.ndarray
        Path to a label file for the surfaces, numpy array containing the
        labels, or a list containing multiple of the aforementioned.
    mask_labels : str, numpy.ndarray, sequence of str of or numpy.ndarray
        Path to a mask file for the surfaces, numpy array containing the
        mask, or a list containing multiple of the aforementioned. If None
        all vertices are included in the mask. Defaults to None.
    interpolation : str, optional
        Either 'nearest' for nearest neighbor interpolation, or 'linear'
        for trilinear interpolation, by default 'linear'.
    data_dir : str, optional
        The directory of the dataset. If none exists, a new dataset will
        be downloaded and saved to this path. If None, the directory defaults to
        your home directory, by default None.


    Returns
    -------
    pandas.DataFrame
        Table with correlation values for each feature.
    """

    data_dir = Path(data_dir) if data_dir else data_directories["NEUROSYNTH_DATA_DIR"]
    data_dir.mkdir(exist_ok=True, parents=True)

    logging.info(
        "Fetching Neurosynth feature files. This may take several minutes if you haven't downloaded them yet."
    )
    feature_files = tuple(_fetch_precomputed(data_dir, database=database))

    mni152 = load_mni152_brain_mask()

    stat_image = tempfile.NamedTemporaryFile(suffix=".nii.gz")
    multi_surface_to_volume(
        pial=pial,
        white=white,
        volume_template=mni152,
        output_file=stat_image.name,
        labels=stat_labels,
        interpolation=interpolation,
    )

    stat_volume = nib.load(stat_image.name)
    mask = (stat_volume.get_fdata() != 0) & (mni152.get_fdata() != 0)
    stat_vector = stat_volume.get_fdata()[mask]

    feature_names = []
    correlations = np.zeros(len(feature_files))

    logging.info("Running correlations with all Neurosynth features.")
    for i in range(len(feature_files)):
        feature_names.append(re.search("__[A-Za-z0-9]+", feature_files[i].stem)[0][2:])  # type: ignore
        feature_data = nib.load(feature_files[i]).get_fdata()[mask]
        keep = np.logical_not(
            np.isnan(feature_data)
            | np.isinf(feature_data)
            | np.isnan(stat_vector)
            | np.isinf(stat_vector)
        )
        correlations[i], _ = pearsonr(stat_vector[keep], feature_data[keep])

    df = pd.DataFrame(correlations, index=feature_names, columns=["Pearson's r"])
    return df.sort_values(by="Pearson's r", ascending=False)


def _fetch_precomputed(data_dir: Path, database: str) -> Generator[Path, None, None]:
    """Wrapper for any future data fetcher.

    Parameters
    ----------
    data_dir : Path
        Directory where the data is stored.
    database : str
        Name of the database, valid arguments are 'neurosynth'.

    Returns
    -------
    generator
        Generator of paths to the precomputed files.

    Raises
    ------
    NotImplementedError
        Returned when requesting the Neuroquery data fetcher.
    ValueError
        Returned when requesting an unknown database.
    """
    if database == "neurosynth":
        return _fetch_precomputed_neurosynth(data_dir)
    elif database == "neuroquery":
        raise NotImplementedError("Neuroquery has not been implemented yet.")
    else:
        raise ValueError(f"Unknown database {database}.")


def _fetch_precomputed_neurosynth(data_dir: Path) -> Generator[Path, None, None]:
    """Downloads precomputed Neurosynth features and returns the filepaths."""

    json = read_data_fetcher_json()["neurosynth_precomputed"]
    url = json["url"]

    existing_files = (data_dir / "upload").glob(
        "Neurosynth_TFIDF__*z_desc-consistency.nii.gz"
    )
    if len(list(existing_files)) != json["n_files"]:
        response = urllib.request.urlopen(url)

        zip_file = tempfile.NamedTemporaryFile(prefix=str(data_dir), suffix=".zip")
        with open(zip_file.name, "wb") as fw:
            fw.write(response.read())

        with zipfile.ZipFile(zip_file.name, "r") as fr:
            fr.extractall(data_dir)

    return data_dir.glob("Neurosynth_TFIDF__*z_desc-consistency.nii.gz")
