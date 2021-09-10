""" Histology context decoder """
import logging
from pathlib import Path
from typing import Callable, Optional, Union

import h5py
import numpy as np
from brainspace.gradient.gradient import GradientMaps
from brainspace.utils.parcellation import reduce_by_labels

from brainstat._typing import ArrayLike
from brainstat._utils import _download_file, data_directories, read_data_fetcher_json


def compute_histology_gradients(
    mpc: np.ndarray,
    kernel: Union[str, Callable[[np.ndarray], np.ndarray]] = "normalized_angle",
    approach: Union[str, Callable[[np.ndarray], np.ndarray]] = "dm",
    n_components: int = 10,
    alignment: Optional[str] = None,
    random_state: Optional[int] = None,
    gamma: Optional[float] = None,
    sparsity: Optional[float] = 0.9,
    reference: Optional[np.ndarray] = None,
    n_iter: int = 10,
) -> GradientMaps:
    """Computes microstructural profile covariance gradients.

    Parameters
    ----------
    mpc : numpy.ndarray
        Microstructural profile covariance matrix.
    kernel : str, optional
         Kernel function to build the affinity matrix. Possible options: {‘pearson’,
         ‘spearman’, ‘cosine’, ‘normalized_angle’, ‘gaussian’}. If callable, must
         receive a 2D array and return a 2D square array. If None, use input matrix.
         By default "normalized_angle".
    approach : str, optional
        Embedding approach. Can be 'pca' for Principal Component Analysis, 'le' for
        laplacian eigenmaps, or 'dm' for diffusion mapping, by default "dm".
    n_components : int, optional
        Number of components to return, by default 10.
    alignment : str, None, optional
        Alignment approach. Only used when two or more datasets are provided.
        Valid options are 'pa' for procrustes analysis and "joint" for joint embedding.
        If None, no alignment is peformed, by default None.
    random_state : int, None, optional
        Random state, by default None
    gamma : float, None, optional
        Inverse kernel width. Only used if kernel == "gaussian". If None, gamma=1/n_feat,
        by default None.
    sparsity : float, optional
        Proportion of smallest elements to zero-out for each row, by default 0.9.
    reference : numpy.ndarray, optional
        Initial reference for procrustes alignments. Only used when
        alignment == 'procrustes', by default None.
    n_iter : int, optional
        Number of iterations for Procrustes alignment, by default 10.

    Returns
    -------
    brainspace.gradient.gradient.GradientMaps
        BrainSpace gradient maps object.
    """
    gm = GradientMaps(
        kernel=kernel,
        approach=approach,
        n_components=n_components,
        alignment=alignment,
        random_state=random_state,
    )
    gm.fit(mpc, gamma=gamma, sparsity=sparsity, n_iter=n_iter, reference=reference)
    return gm


def compute_mpc(profile: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Computes MPC for given labels on a surface template.

    Parameters
    ----------
    profile : numpy.ndarray
        Histological profiles of size surface-by-vertex.
    labels : numpy.ndarray
        Labels of regions of interest. Use 0 to denote regions that will not be included.

    Returns
    -------
    numpy.ndarray
        Microstructural profile covariance.
    """

    roi_profile = reduce_by_labels(profile, labels)
    if np.any(labels == 0):
        # Remove 0's in the labels.
        roi_profile = roi_profile[:, 1:]

    p_corr = partial_correlation(roi_profile, np.mean(profile, axis=1))

    mpc = 0.5 * np.log((1 + p_corr) / (1 - p_corr))
    mpc[p_corr > 0.99999] = 0  # Deals with floating point issues where p_corr==1
    mpc[mpc == np.inf] = 0
    mpc[mpc == np.nan] = 0

    return mpc


def read_histology_profile(
    data_dir: Optional[Union[str, Path]] = None,
    template: str = "fsaverage",
    overwrite: bool = False,
) -> np.ndarray:
    """Reads BigBrain histology profiles.

    Parameters
    ----------
    data_dir : str, pathlib.Path, None, optional
        Path to the data directory. If data is not found here then data will be
        downloaded. If None, data_dir is set to the home directory, by default None.
    template : str, optional
        Surface template. Currently allowed options are 'fsaverage' and 'fslr32k', by
        default 'fsaverage'.
    overwrite : bool, optional
        If true, existing data will be overwrriten, by default False.

    Returns
    -------
    numpy.ndarray
        Depth-by-vertex array of BigBrain intensities.
    """

    data_dir = Path(data_dir) if data_dir else data_directories["BIGBRAIN_DATA_DIR"]
    histology_file = data_dir / ("histology_" + template + ".h5")

    if not histology_file.exists() or overwrite:
        logging.info(
            "Could not find a histological profile or an overwrite was requested. Downloading..."
        )
        download_histology_profiles(
            data_dir=data_dir, template=template, overwrite=overwrite
        )

    with h5py.File(histology_file, "r") as h5_file:
        return h5_file.get(template)[...]


def download_histology_profiles(
    data_dir: Optional[Union[str, Path]] = None,
    template: str = "fsaverage",
    overwrite: bool = False,
) -> None:
    """Downloads BigBrain histology profiles.

    Parameters
    ----------
    data_dir : str, pathlib,Path, None, optional
        Path to the directory to store the data. If None, defaults to the home
        directory, by default None.
    template : str, optional
        Surface template. Currently allowed options are 'fsaverage' and 'fslr32k', by
        default 'fsaverage'.
    overwrite : bool, optional
        If true, existing data will be overwrriten, by default False.

    Raises
    ------
    KeyError
        Thrown if an invalid template is requested.
    """

    data_dir = Path(data_dir) if data_dir else data_directories["BIGBRAIN_DATA_DIR"]
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / ("histology_" + template + ".h5")

    url = read_data_fetcher_json()["bigbrain_profiles"][template]

    try:
        _download_file(url, output_file, overwrite)
    except KeyError:
        raise KeyError(
            "Could not find the requested template. Valid templates are: 'fslr32k', 'fsaverage', 'fsaverage5'."
        )


def partial_correlation(X: ArrayLike, covar: np.ndarray) -> np.ndarray:
    """Runs a partial correlation whilst correcting for a covariate.

    Parameters
    ----------
    X : ArrayLike
        Two-dimensional array of the data to be correlated.
    covar : numpy.ndarray
        One-dimensional array of the covariate.

    Returns
    -------
    numpy.ndarray
        Partial correlation matrix.
    """
    X_mean = np.concatenate((X, covar[:, None]), axis=1)
    pearson_correlation = np.corrcoef(X_mean, rowvar=False)
    r_xy = pearson_correlation[:-1, :-1]
    r_xz = pearson_correlation[0:-1, -1][:, None]

    return (r_xy - r_xz @ r_xz.T) / (np.sqrt(1 - r_xz ** 2) * np.sqrt(1 - r_xz.T ** 2))
