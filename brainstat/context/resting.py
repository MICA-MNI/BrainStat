from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from brainspace.utils.parcellation import reduce_by_labels

from brainstat._typing import ArrayLike
from brainstat.datasets import fetch_gradients, fetch_parcellation


def yeo_networks_associations(
    data: ArrayLike,
    template: str = "fsaverage5",
    seven_networks: bool = True,
    data_dir: Optional[Union[str, Path]] = None,
    reduction_operation: Union[str, Callable] = np.nanmean,
) -> np.ndarray:
    """Computes association

    Parameters
    ----------
    data : ArrayLike
        Data to be summarized in the Yeo networks in a sample-by-feature format.
    template : str, optional
        Surface template. Valid values are "fsaverage5", "fsaverage", and
        "fslr32k", "civet41k", and "civet164k", by default "fsaverage5".
    seven_networks : bool, optional
        If true, uses the 7 network parcellation, otherwise uses the 17 network
        parcellation, by default True.
    data_dir : str, Path, optional
        Data directory to store the Yeo network files, by default $HOME_DIR/brainstat_data/parcellation_data.
    reduction_operation : str, callable, optional
        How to summarize data. If str, options are: {‘min’, ‘max’, ‘sum’,
        ‘mean’, ‘median’, ‘mode’, ‘average’}. If callable, it should receive a
        1D array of values, array of weights (or None) and return a scalar
        value. Default is ‘mean’.

    Returns
    -------
    np.ndarray
        Summary statistic in the yeo networks.
    """
    n_regions = 7 if seven_networks else 17
    yeo_networks = fetch_parcellation(
        template=template,
        atlas="yeo",
        n_regions=n_regions,
        join=True,
        data_dir=data_dir,
    )

    if np.array(data).ndim == 1:
        data_2d = np.array(data)[:, None]
    else:
        data_2d = np.array(data)

    n_features = data_2d.shape[1]

    yeo_mean = np.zeros((n_regions + 1, n_features))
    for i in range(n_features):
        yeo_mean[:, i] = reduce_by_labels(
            data_2d[:, i], yeo_networks, red_op=reduction_operation
        )
    return yeo_mean[1:, :]


def gradients_corr(
    data: ArrayLike,
    name: str = "margulies2016",
    template: str = "fsaverage5",
    data_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> np.ndarray:
    """Comptues the correlation of the input data with the Margulies gradients.

    Parameters
    ----------
    data : ArrayLike
        The data to be compared to the Margulies gradients. Data must be in the
        shape of vertices-by-features.
    name : str, optional
        Name of the gradients. Valid values are "margulies2016", defaults to
        "margulies2016".
    template : str, optional
        Name of the template surface. Valid values are "fsaverage5",
        "fsaverage7", "fslr32k", defaults to "fsaverage5".
    data_dir : str, Path, optional
        Path to the directory to store the Margulies gradient data files, by
        default $HOME_DIR/brainstat_data/functional_data.
    overwrite : bool, optional
        If true, overwrites existing files, by default False.

    Returns
    -------
    np.ndarray
        Correlations between the input data and the Margulies gradients.
    """
    gradients = fetch_gradients(
        name=name, template=template, data_dir=data_dir, overwrite=overwrite
    )
    return _columnwise_correlate(data, gradients)


def _columnwise_correlate(x: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
    """Implements MATLAB's corr function for Pearson correlations of each column
    in x with each column in y. If y is not provided, computes correlation with
    x onto itself.

    Parameters
    ----------
    x : ArrayLike
        2D data matrix.
    y : ArrayLike, optional
        2D data matrix.

    Implements the function R = (1/n-1) @ X_s' @ Y_s.
    - n is the number of samples.
    - X_s and Y_s are standardized versions of X and Y i.e. X_s = C @ X @ D.
    - C is the centering matrix.
    - D is a scaling matrix i.e. a diagonal matrix with std(x, axis=0) on the
        diagonal.

    Returns
    -------
    np.ndarray
        Pearson correlation matrix.
    """

    n_samples, n_features = np.shape(x)
    centering = np.identity(n_samples) - 1 / n_samples
    scaling = lambda v: np.identity(n_features) * np.std(v, axis=0, ddof=1)
    centered_scaled = lambda v: centering @ v @ np.linalg.inv(scaling(v))

    if y is None:
        X_s = centered_scaled(x)
        return 1 / (n_samples - 1) * X_s.T @ X_s
    else:
        return 1 / (n_samples - 1) * centered_scaled(x).T @ centered_scaled(y)
