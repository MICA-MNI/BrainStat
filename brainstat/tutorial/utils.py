import warnings
from pathlib import Path

import pandas as pd
from nilearn.datasets.utils import _fetch_files, _get_dataset_dir
from sklearn.utils import Bunch


def fetch_tutorial_data(n_subjects=20, data_dir=None, resume=True, verbose=1):

    """Download and load the surfstat tutorial dataset.

    Parameters
    ----------
    n_subjects: int, optional
        The number of subjects to load from maximum of 100 subjects.
        By default, 20 subjects will be loaded. If None is given,
        all 100 subjects will be loaded.
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. If None, data will be download to ~ (home directory).
        Default: None
    resume: bool, optional
        If true, try resuming download if possible

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
         - 'image_files': Paths to image files in mgh format
         - 'demographics': Path to CSV file containing demographic information

    References
    ----------

    :Download: https://box.bic.mni.mcgill.ca/s/wMPF2vj7EoYWELV

    """

    # set dataset url
    url = "https://box.bic.mni.mcgill.ca/s/wMPF2vj7EoYWELV"

    # set data_dir, if not directly set use ~ as default
    if data_dir is None:
        data_dir = str(Path.home())

    # set dataset name and get its corresponding directory
    dataset_name = "brainstat_tutorial"
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    # set download information for demographic file
    files = [
        (
            "brainstat_tutorial_df.csv",
            url + "/download?path=%2FSurfStat_tutorial_data&files=myStudy.csv",
            {"move": "brainstat_tutorial_df.csv"},
        )
    ]

    # download demographic file
    path_to_demographics = _fetch_files(data_dir, files, verbose=verbose)[0]

    # set ids based on complete dataset from demographic file
    ids = pd.read_csv(path_to_demographics)["ID2"].tolist()

    # set and check subjects, in total and subset
    max_subjects = len(ids)
    if n_subjects is None:
        n_subjects = max_subjects
    if n_subjects > max_subjects:
        warnings.warn("Warning: there are only %d subjects" % max_subjects)
        n_subjects = max_subjects
    ids = ids[:n_subjects]

    # restrict demographic information to subset of subjects
    df_tmp = pd.read_csv(path_to_demographics)
    df_tmp = df_tmp[df_tmp["ID2"].isin(ids)]

    # set download information for image files and download them
    # for hemi in ['lh', 'rh']:
    image_files = _fetch_files(
        data_dir,
        [
            (
                "thickness/{}_{}2fsaverage5_20.mgh".format(subj, hemi),
                url + "/download?path=%2F&files=brainstat_tutorial.zip",
                {"uncompress": True, "move": "brainstat_tutorial.zip"},
            )
            for subj in ids
            for hemi in ["lh", "rh"]
        ],
    )

    # pack everything in a scikit-learn bunch and return it
    return Bunch(demographics=df_tmp, image_files=image_files)
