from scipy.interpolate import interp1d
from itertools import product
import numpy as np 


def row_ismember(a, b):
    # a is 2D numpy array
    # b is 2D numpy array
    # returns the adress of a-rows in b-rows (if they are identical)
    # much faster than ismember function below
    bind = {}
    for i, elt in enumerate(b):
        if tuple(elt) not in bind:
            bind[tuple(elt)] = i
    return [bind.get(tuple(itm), None) for itm in a] 

def interp1(x,y,ix,kind='linear'):
    # Let's call it interp1_mp like matlab-python
    # Shorthand for the MATLAB interp1 function.  
    f = interp1d(x, y, kind)
    return f(ix)

def ismember(A, B, rows=False):
    # Implementation of MATLAB's ismember() function. 
    # Tests if elements of A appear in B. Returns a logical array and a vector
    # containing the index of the first appearance of each member.
    # Note for the locations output variable: MATLAB uses 0 to denote missing values.
    # As 0 is a valid location in Python, we use NaN here instead. 
    
    if rows:
        # Get rows of A that are in B.
        equality = np.equal(np.expand_dims(A,axis=2), np.expand_dims(B.T,axis=0))
        equal_rows = np.squeeze(np.all(equality,axis=1))
        bool_array = np.any(equal_rows,1)

        # Get location of elements in B.
        locations = np.zeros(bool_array.shape) + np.nan 
        for i in range(0,equal_rows.shape[0]):
            nz = np.nonzero(equal_rows[i,:])
            if nz[0].size != 0:
                locations[i] = nz[0]

    else:
        # Get values of A that are in B.
        bool_vector = np.in1d(A, B)
        bool_array = np.reshape(bool_vector,A.shape)

        # Get location of elements in B. Transpose B and A to get MATLAB behavior (i.e. column first)
        val, locB = np.unique(B.T,return_index=True)
        idx = np.flatnonzero(bool_array) 
        locations = np.zeros(A.size) + np.nan
        Aflat = A.T.flat
        for i in range(0,idx.size):
            locations[idx[i]] = locB[np.argwhere(val==Aflat[idx[i]])]
        locations = np.reshape(locations,A.shape)
    return bool_array, locations

def colon(start,stop,increment=1):
    """ Generates a range of numbers including the stop number. 
    Parameters
    ----------
    start : starting scalar number of the range.
    stop : stopping scalar number of the range.
    increment : increments of the range
    
    Returns
    -------
    r : Numpy array of the requested numbers.  
    """
    r = np.arange(start,stop,increment)
    if start > stop:
        return r
    elif start == stop or r[-1] + increment == stop:
        r = np.append(r,stop)
    return r

def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
    """
    An accumulation function similar to Matlab's `accumarray` function.
    Function downloaded from https://scipy.github.io/old-wiki/pages/Cookbook/AccumarrayLike.html
    
    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1) 
    a : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array. 
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.
    Returns
    -------
    out : ndarray
        The accumulated results.
        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.
    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = accum(accmap, a)
    array([9, 7, 15])
    >>> # A 2D output, from sub-arrays with shapes and positions like this:
    >>> # [ (2,2) (2,1)]
    >>> # [ (1,2) (1,1)]
    >>> accmap = array([
            [[0,0],[0,0],[0,1]],
            [[0,0],[0,0],[0,1]],
            [[1,0],[1,0],[1,1]],
        ])
    >>> # Accumulate using a product.
    >>> accum(accmap, a, func=prod, dtype=float)
    array([[ -8.,  18.],
           [ -8.,   9.]])
    >>> # Same accmap, but create an array of lists of values.
    >>> accum(accmap, a, func=lambda x: x, dtype='O')
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)
    """

    # Check for bad arguments and handle the defaults.
    if accmap.shape[:a.ndim] != a.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = a.dtype
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in a.shape]):
        indx = tuple(accmap[s])
        val = a[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])

    return out