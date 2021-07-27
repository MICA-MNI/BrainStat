"""Utilities for the stats functions."""

import warnings

import numpy as np
from scipy.interpolate import interp1d


def row_ismember(a, b):
    """Tests whether rows of a occur in b.

    Parameters
    ----------
    a : numpy.array
        a 2D array with the same number of columns as b.
    b : numpy.array
        a 2D array with the same number of columns as a.

    Returns
    -------
    numpy.array
        Indices of rows in a that occur in b.
    """

    bind = {}
    for i, elt in enumerate(b):
        if tuple(elt) not in bind:
            bind[tuple(elt)] = i
    return [bind.get(tuple(itm), None) for itm in a]


def interp1(x, y, ix, kind="linear"):
    """Interpolation between datapoints.

    Parameters
    ----------
    x : (numpy.array)
        x coordinates of training data.
    y : (numpy.array)
        y coordinates of training data.
    ix : (numpy.array)
        coordinates of the interpolated points.
    kind (numpy.array)
        type of interpolation; see scipy.interpolate.interp1d for options.

    Returns
    -------
    numpy.array
        interpolated y coordinates.
    """

    f = interp1d(x, y, kind, bounds_error=False, fill_value=np.nan)
    iy = f(ix)
    return iy


def ismember(A, B, rows=False):
    """Tests whether elements of A appear in B.

    Parameters
    ----------
    A : (numpy.array)
        1D or 2D array
    B : (numpy.array)
        1D or 2D array
    rows : (logical)
        If true test for row correspondence rather than element correpondence.

    Returns
    -------
    logical
        Boolean of the same size as A denoting which elements (or rows) occur in B.
    numpy.array
        Indices of matching elements/rows in A.

    Notes
    -----
    For row-wise comparisons, row_ismember should be significantly faster.

    """

    if rows:
        # Get rows of A that are in B.
        equality = np.equal(np.expand_dims(A, axis=2), np.expand_dims(B.T, axis=0))
        equal_rows = np.squeeze(np.all(equality, axis=1))
        bool_array = np.any(equal_rows, 1)

        # Get location of elements in B.
        locations = np.zeros(bool_array.shape) + np.nan
        for i in range(0, equal_rows.shape[0]):
            nz = np.nonzero(equal_rows[i, :])
            if nz[0].size != 0:
                locations[i] = nz[0]

    else:
        # Get values of A that are in B.
        bool_vector = np.in1d(A, B)
        bool_array = np.reshape(bool_vector, A.shape)

        # Get location of elements in B. Transpose B and A to get MATLAB behavior (i.e. column first)
        val, locB = np.unique(B.T, return_index=True)
        idx = np.flatnonzero(bool_array)
        locations = np.zeros(A.size) + np.nan
        Aflat = A.T.flat
        for i in range(0, idx.size):
            locations[idx[i]] = locB[np.argwhere(val == Aflat[idx[i]])]
        locations = np.reshape(locations, A.shape)
    return bool_array, locations


def colon(start, stop, increment=1):
    """Generates a range of numbers including the stop number.

    Parameters
    ----------
    start : (int)
        Starting scalar number of the range.
    stop :  (int)
        Stopping scalar number of the range.
    increment (float)
        Increments of the range.

    Returns
    -------
    numpy.array
        The requested numbers.
    """
    r = np.arange(start, stop, increment)
    if (start > stop and increment > 0) or (start < stop and increment < 0):
        return r
    elif start == stop or r[-1] + increment == stop:
        r = np.append(r, stop)
    return r


def apply_mask(Y, mask, axis=0):
    """Masks the data along a specified axis

    Parameters
    ----------
    Y : array-like
        Data to be masked.
    mask : array-like
        Boolean vector containing True for each element to keep.
    axis : int, optional
        Axis along which to operate, by default 0.

    Returns
    -------
    numpy.array
        Masked data.
    """
    Y = Y.swapaxes(0, axis)
    Y = Y[mask, ...]
    return Y.swapaxes(0, axis)


def undo_mask(Y, mask, axis=0, missing_value=np.nan):
    """Restores the original dimensions of masked data.

    Parameters
    ----------
    Y : array-like
        Masked data.
    mask : array-like
        Boolean vector used to mask the data.
    axis : int, optional
        Axis along which to operate, by default 0.
    missing_value : scalar, optional
        Number to insert for missing values, by default np.nan.

    Returns
    -------
    numpy.array
        Unmasked data.
    """
    new_dims = list(Y.shape)
    new_dims[axis] = mask.size
    Y2 = np.empty(new_dims)
    Y2[:] = missing_value

    Y = Y.swapaxes(0, axis)
    Y2 = Y2.swapaxes(0, axis)
    Y2[mask, ...] = Y
    Y2 = Y2.swapaxes(0, axis)

    return Y2


def deprecated(message):
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
