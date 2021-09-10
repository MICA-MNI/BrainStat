import shutil
import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from tqdm import tqdm

from brainstat._utils import data_directories, read_data_fetcher_json


def fetch_abide_data(
    data_dir: Optional[Union[str, Path]] = None,
    sites: Sequence[str] = None,
    keep_control: bool = True,
    keep_patient: bool = True,
    overwrite: bool = False,
) -> Tuple[Bunch, Bunch]:

    data_dir = Path(data_dir) if data_dir else data_directories["ABIDE_DATA_DIR"]
    data_dir.mkdir(exist_ok=True, parents=True)
    summary_spreadsheet = data_dir / "summary_spreadsheet.csv"

    if not summary_spreadsheet.is_file():
        summary_url = read_data_fetcher_json()["abide_tutorial"]["summary_spreadsheet"]
        _download_file(summary_spreadsheet, summary_url["url"])

    df = pd.read_csv(summary_spreadsheet)
    _select_subjects(df, sites, keep_patient, keep_control)

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
                    _download_file(filename, thickness_url)
                except HTTPError:
                    warnings.warn(f"Could not download file for {row.SUB_ID}.")
                    remove_rows.append(i)
                    continue

            thickness_data[i, j * 40962 : (j + 1) * 40962] = np.loadtxt(filename)

    if remove_rows:
        thickness_data = np.delete(thickness_data, remove_rows, axis=0)
        df.drop(np.unique(remove_rows), inplace=True)
        df.reset_index(inplace=True)

    return thickness_data, df


def _download_file(filename: Path, url: str) -> None:
    if not filename.is_file():
        with urlopen(url) as u, open(filename, "wb") as f:
            shutil.copyfileobj(u, f)


def _select_subjects(
    df: pd.DataFrame,
    sites: Optional[Sequence[str]],
    keep_patient: bool,
    keep_control: bool,
) -> None:
    df.drop(df[df.FILE_ID == "no_filename"].index, inplace=True)
    if not keep_patient:
        df.drop(df[df.DX_GROUP == 1].index, inplace=True)

    if not keep_control:
        df.drop(df[df.DX_GROUP == 2].index, inplace=True)

    if sites is not None:
        df.drop(df[~df.SITE_ID.isin(sites)].index, inplace=True)

    df.reset_index(inplace=True)
