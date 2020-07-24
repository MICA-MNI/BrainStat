import warnings
from pathlib import Path
import pandas as pd
from sklearn.utils import Bunch
from nilearn.datasets.utils import (_get_dataset_dir, _fetch_files)


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
        location. If None, data will be download to ~ (home directory). Default: None
    resume: bool, optional
        If true, try resuming download if possible

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
         - 'surface_files': Paths to surface files in mgh format
         - 'demographics': Path to CSV file containing demographic information

    References
    ----------

    :Download: https://box.bic.mni.mcgill.ca/s/wMPF2vj7EoYWELV

    """

    url = "https://box.bic.mni.mcgill.ca/s/wMPF2vj7EoYWELV"

    if data_dir is None:
        data_dir = str(Path.home())

    dataset_name = "brainstat_tutorial"

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    files = [
        (
            "brainstat_tutorial_df.csv",
            url + "/download?path=%2FSurfStat_tutorial_data&files=myStudy.csv",
            {"move": "brainstat_tutorial_df.csv"},
        )
    ]
    path_to_demographics = _fetch_files(data_dir, files, verbose=verbose)[0]


    ids = pd.read_csv(path_to_demographics)["ID2"].tolist()

    max_subjects = len(ids)
    if n_subjects is None:
        n_subjects = max_subjects
    if n_subjects > max_subjects:
        warnings.warn('Warning: there are only %d subjects' % max_subjects)
        n_subjects = max_subjects
    ids = ids[:n_subjects]

    for hemi in ['lh', 'rh']:
        surface_files = _fetch_files(data_dir, [('{}_{}2fsaverage5_20.mgh'.format(subj, hemi),
                                                 url + "/download?path=%2FSurfStat_tutorial_data%2Fthickness&files={}_{}2fsaverage5_20.mgh".format(subj, hemi),
                                                {'move': '{}_{}2fsaverage5_20.mgh'.format(subj, hemi)}) for subj in ids])

    return Bunch(demographics=path_to_demographics,
                 surface_files=surface_files)



