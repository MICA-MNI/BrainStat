"""
python version of SurfStatPCA
"""

# Author: RRC
# License: BSD 3 clause


import numpy as np

def py_SurfStatPCA(Y, mask, X, k):
    """Principal Components Analysis (PCA).
    Parameters
    ----------
    Y    = n x v matrix or n x v x k array of data, v=#vertices,
           or memory map of same.
    mask = 1 x v vector, 1=inside, 0=outside, default is ones(1,v),
           i.e. the whole surface.
    X    = model formula of type term, or scalar, or n x p design matrix of
           p covariates for the linear model. The PCA is done on the v x v
           correlations of the residuals and the components are standardized
           to have unit standard deviation about zero. If X=0, nothing is
           removed. If X=1, the mean (over rows) is removed (default).
    c    = number of components in PCA, default 4.
    %
    pcntvar = 1 x c vector of percent variance explained by the components.
    U       = n x c matrix of components for the rows (observations).
    V       = c x v x k array of components for the columns (vertices).

    Returns
    -------
    source : 2D ndarray, shape = (n_samples, n_feat)
        Source dataset.
    target : 2D ndarray, shape = (n_samples, n_feat)
        Target dataset.
    center : bool, optional
        Center data before alignment. Default is False.
    scale : bool, optional
        Remove scale before alignment. Default is False.
    """

    sys.exit("Function py_SurfStatPCA is not implemented yet")
