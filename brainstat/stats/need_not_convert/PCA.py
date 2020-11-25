import sys
import numpy as np
import numpy.matlib
from term import Term
import matlab.engine
import matlab
global eng
eng = matlab.engine.start_matlab()
addpath = eng.addpath('../matlab')


def py_SurfStatPCA(Y, mask=None, X=1, c=4):
    """Principal Components Analysis (PCA).

    Parameters
    ----------
    Y : 2D numpy array of shape (n,v) , or 3D numpy array of shape (n,v,k),
        v is the number of vertices.
    mask : 2D numpy array of shape (1,v), ones and zeros,
        mask array, 1=inside, 0=outside, by default np.ones((1,v)).
    X : a scalar, or 2D numpy array of shape (n,p), or type term,
        if array, X is a design matrix of p covariates for the linear model,
        if term, X is model formula. The PCA is done on the v x v correlations
        of the residuals and the components are standardized to have unit
        standard deviation about zero. If X=0, nothing is removed. If X=1,
        the mean (over rows) is removed (default).
    c : int (<= n), by default 4,
        number of components in PCA

    Returns
    -------
    pcntvar : 2D numpy array of shape (1,c),
        array of percent variance explained by the components.
    U : 2D numpy array of shape (n,c),
        array of components for the rows (observations).
    V : 2D numpy array of shape (c,v) or 3D numpy array of shape (c,v,k),
        array of components for the columns (vertices).
    """

    if Y.ndim == 2:
        n, v = Y.shape
        k = 1
    elif Y.ndim > 2:
        n, v, k = Y.shape

    if mask is None:
        mask = np.ones((1,v))

    if np.isscalar(X):
        X = np.matlib.repmat(X,n,1)
    elif isinstance(X, np.ndarray):
        if np.shape(X)[0] == 1:
            X = np.matlib.repmat(X,n,1)
    elif isinstance(X, Term):
        X = X.matrix.values.T
        if np.shape(X)[0] == 1:
            X = np.matlib.repmat(X,n,1)

    df = n - np.linalg.matrix_rank(X)
    nc = 1
    chunk = v
    A = np.zeros((n,n))

    for ic in np.arange(1, nc+1, 1):
        v1 = 1 + (ic - 1)*chunk
        v2 = min(v1 + chunk - 1, v)
        vc = v2 - v1 + 1
        maskc = mask[v1 - 1 : v2]

        if k == 1:
            Y = Y[:, (maskc-1).astype(int).tolist()[0]]
        else:
            Y = np.reshape(Y[:, (maskc-1).astype(int).tolist()[0], :],
                             (n, int(maskc.sum()*k)))

        if np.any(X[:] != 0):
            Y = Y - X @ (np.linalg.pinv(X) @ Y)

        Y = Y.astype(float)
        S = np.sum(Y**2, axis=0).astype(float)
        Smhalf = (S>0) / np.sqrt(S + (S<=0))

        for i in np.arange(1, n+1, 1):
            Y[i-1,:] = Y[i-1,:] * Smhalf

        A = A + Y @ Y.T


    #D, U = np.linalg.eig(A)  #### matlab part differs!!!
    #D = np.diag(D)

    # matlab part still differs
    U, D = eng.eig(matlab.double(A.T.tolist()), nargout=2)
    D = np.array(D)
    U = np.array(U)

    ds = np.sort(np.diag(-D))
    iss = np.argsort(np.diag(-D))
    ds = -ds
    pcntvar = ds[0:c].reshape(1,-1) / ds.sum()*100
    U = U[:, iss[0:c]]

    V = np.zeros((c, v*k))
    V[:, (np.matlib.repmat(mask,1,k)-1).astype(int).tolist()[0]] = U.T @ Y

    s = np.sign(abs(V.max(1)) - abs(V.min(1)))
    sv = np.sqrt(np.mean(V**2, axis=1))

    V = np.diag(s/(sv + (sv<=0))*(sv>0)) @ V
    U = U @ np.diag(s * np.sqrt(df) )

    if k > 1:
        V = V.reshape(c,v,k)

    return pcntvar, U, V
