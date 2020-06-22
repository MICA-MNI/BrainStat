from scipy.interpolate import interp1d
import numpy as np 

def interp1(x,y,ix):
    # Shorthand for the MATLAB interp1 function.  
    f = interp1d(x, y)
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
    # Implementation of the matlab colon() function (e.g. a=1:5;).
    # Essentially it's np.arange but includes the endpoint.
    m = np.arange(start,stop,increment)
    if start == stop or m[-1] + increment == stop:
        m = np.append(m,stop)
    return m