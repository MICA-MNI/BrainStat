from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from urllib.error import HTTPError

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainstat._utils import (
    _download_file,
    data_directories,
    logger,
    read_data_fetcher_json,
)


def fetch_mics_data(
    data_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fetches MICS cortical thickness data.

    Parameters
    ----------
    data_dir : str, pathlib.Path, optional
        Path to store the MICS data, by default $HOME_DIR/brainstat_data/mics_data.
    overwrite : bool, optional
        If true overwrites existing data, by default False

    Returns
    -------
    np.ndarray
        Subject-by-vertex cortical thickness data on fsaverage5.
    pd.DataFrame
        Subject demographics.
    """

    data_dir = Path(data_dir) if data_dir else data_directories["MICS_DATA_DIR"]
    data_dir.mkdir(exist_ok=True, parents=True)
    demographics_file = data_dir / "mics_demographics.csv"

    demographics_url = read_data_fetcher_json()["mics_tutorial"]["demographics"]
    _download_file(demographics_url["url"], demographics_file, overwrite)
    df = pd.read_csv(demographics_file)

    thickness_file = data_dir / "mics_thickness.h5"
    thickness_url = read_data_fetcher_json()["mics_tutorial"]["thickness"]
    _download_file(thickness_url["url"], thickness_file, overwrite)

    with h5py.File(thickness_file, "r") as f:
        thickness = f["thickness"][:]

    return thickness, df


def fetch_abide_data(
    data_dir: Optional[Union[str, Path]] = None,
    sites: Sequence[str] = None,
    keep_control: bool = True,
    keep_patient: bool = True,
    overwrite: bool = False,
    min_rater_ok: int = 3,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fetches ABIDE cortical thickness data.

    Parameters
    ----------
    data_dir : str, pathlib.Path, optional
        Path to store the MICS data, by default $HOME_DIR/brainstat_data/mics_data.
    sites : list, tuple, optional
        List of sites to include. If none, uses all sites, by default None.
    keep_control : bool, optional
        If true keeps control subjects, by default True.
    keep_patient : bool, optional
        If true keeps patient subjects, by default True.
    overwrite : bool, optional
        If true overwrites existing data, by default False.
    min_rater_ok : int, optional
        Minimum number of raters who approved the data, by default 3.

    Returns
    -------
    np.ndarray
        Subject-by-vertex cortical thickness data on fsaverage5.
    pd.DataFrame
        Subject demographics.
    """
    data_dir = Path(data_dir) if data_dir else data_directories["ABIDE_DATA_DIR"]
    data_dir.mkdir(exist_ok=True, parents=True)
    summary_spreadsheet = data_dir / "summary_spreadsheet.csv"

    summary_url = read_data_fetcher_json()["abide_tutorial"]["summary_spreadsheet"]
    _download_file(summary_url["url"], summary_spreadsheet, overwrite)
    df = pd.read_csv(summary_spreadsheet)
    _select_subjects(df, sites, keep_patient, keep_control, min_rater_ok)

    # Download subject thickeness data
    def _thickness_url(derivative, identifier):
        return f"https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/civet/thickness_{derivative}/{identifier}_{derivative}.txt"

    thickness_data = np.zeros((df.shape[0], 81924))
    remove_rows = []
    progress_bar = tqdm(df.itertuples())
    for i, row in enumerate(progress_bar):
        progress_bar.set_description(
            f"Fetching thickness data for subject {i+1} out of {df.shape[0]}"
        )
        for j, hemi in enumerate(["left", "right"]):
            filename = data_dir / f"{row.SUB_ID}_{hemi}_thickness.txt"
            if not filename.is_file() or overwrite:
                thickness_url = _thickness_url(
                    f"native_rms_rsl_tlink_30mm_{hemi}", row.FILE_ID
                )
                try:
                    _download_file(thickness_url, filename, overwrite, verbose=False)
                except HTTPError:
                    logger.warn(f"Could not download file for {row.SUB_ID}.")
                    remove_rows.append(i)
                    continue

            thickness_data[i, j * 40962 : (j + 1) * 40962] = np.loadtxt(filename)

    if remove_rows:
        thickness_data = np.delete(thickness_data, remove_rows, axis=0)
        df.drop(np.unique(remove_rows), inplace=True)
        df.reset_index(inplace=True)

    return thickness_data, df


def _select_subjects(
    df: pd.DataFrame,
    sites: Optional[Sequence[str]],
    keep_patient: bool,
    keep_control: bool,
    min_rater_ok: int,
) -> None:
    """Selects subjects based on demographics and site."""
    df.drop(df[df.FILE_ID == "no_filename"].index, inplace=True)
    if not keep_patient:
        df.drop(df[df.DX_GROUP == 1].index, inplace=True)

    if not keep_control:
        df.drop(df[df.DX_GROUP == 2].index, inplace=True)

    if sites is not None:
        df.drop(df[~df.SITE_ID.isin(sites)].index, inplace=True)

    if min_rater_ok > 0:
        rater_approved = (
            df[["qc_rater_1", "qc_anat_rater_2", "qc_anat_rater_3"]]
            .eq("OK")
            .sum(axis=1)
        )
        df.drop(df[rater_approved < min_rater_ok].index, inplace=True)

    df.reset_index(inplace=True)
