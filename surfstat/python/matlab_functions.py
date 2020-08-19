from scipy.interpolate import interp1d
from itertools import product
import numpy as np 

def interp1(x,y,ix,kind='linear'):
    """ Interpolation between datapoints.
    Parameters
    ----------
    x : x coordinates of training data. 
    y : y coordinates of training data.
    ix : x coordinates of the interpolated points. 
    kind : type of interpolation; see scipy.interpolate.interp1d for options. 
    
    Returns
    -------
    iy : interpolated y coordinates.   
    """ 
    
    f = interp1d(x, y, kind)
    iy = f(ix)
    return iy

def ismember(A, B, rows=False):
    """ Tests whether elements of A appear in B.
    Parameters
    ----------
    A : 1D or 2D numpy array 
    B : 1D or 2D numpy array 
    rows : logical denoting whether to test for element-wise occurence or row occurence. 
    
    Returns
    -------
    bool_array : Boolean of the same size as A denoting which elements (or rows) occur in B.
    locations : Indices of matching elements/rows in A.  
    """ 
    
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
        locations = [int(round(x)) for x in locations]
    locations = [int(round(x)) for x in locations]
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