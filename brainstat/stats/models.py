"""
test docstring
"""
import warnings
import numpy as np
from numpy import concatenate as cat
import numpy.linalg as la
import scipy
from scipy.linalg import null_space
from scipy.linalg import cholesky
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix
import math
from cmath import sqrt
import copy
from .utils import interp1, ismember, row_ismember
from ..mesh.utils import mesh_edges
from .terms import Term, Random
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from brainspace.mesh.mesh_elements import get_cells


def f_test(slm1, slm2):
    """ F-statistics for comparing two uni- or nulti-variate fixed effects models.
    Parameters
    ----------
    slm1 : a dictionary with keys 'X', 'df', 'SSE' and 'coef'
        slm1['X'] : 2D numpy array of shape (n,p), the design matrix.
        surf['df'] : int, degrees of freedom.
        surf['SSE'] : 2D numpy array of shape (k*(k+1)/2,v), sum of squares of errors.
        surf['coef'] : 2D or 3D numpy array of shape (p,v) or (p,v,k)
    slm2 : the same style as slm1

    Returns
    -------
    slm : a dictionary with keys 'X', 'df', 'SSE', 'coef', 'k' and 't'
        slm['X'], slm['SSE'], slm['coef']: copied from the bigger model (slm1 or slm2)
        slm['df'] : 2D numpy array of shape (1,2), it is equal to [df1-df2, df2],
                    where df1 and df2 are the min and max of slm1.df and slm2.df,
                    and SSE1 and SSE2 are the corresponding slm_.SSE's.
        slm['k'] : k, the number of variates.
        slm['t'] : 2D numpy array of shape (l,v), non-zero eigenvalues in descending
                   order, of F = (SSE1-SSE2)/(df1-df2)/(SSE2/df2), where
                   l=min(k,df1-df2);  slm.t(1,:) = Roy's maximum root = maximum F
                   over all linear combinations of the k variates.
                   k>3 is not programmed yet.
    """

    if 'r' in slm1.keys() or 'r' in slm2.keys():
        warnings.warn("Mixed effects models not programmed yet.")

    if slm1['df'] > slm2['df']:
        X1 = slm1['X']
        X2 = slm2['X']
        df1 = slm1['df']
        df2 = slm2['df']
        SSE1 = slm1['SSE']
        SSE2 = slm2['SSE']
        slm = slm2.copy()
    else:
        X1 = slm2['X']
        X2 = slm1['X']
        df1 = slm2['df']
        df2 = slm1['df']
        SSE1 = slm2['SSE']
        SSE2 = slm1['SSE']
        slm = slm1.copy()

    r = X1 - np.dot(np.dot(X2, np.linalg.pinv(X2)), X1)
    d = np.sum(r.flatten()**2) / np.sum(X1.flatten()**2)

    if d > np.spacing(1):
        print('Models are not nested.')
        return

    slm['df'] = np.array([[df1-df2, df2]])
    h = SSE1 - SSE2

    # if slm['coef'] is 3D and third dimension is 1, then squeeze it to 2D
    if np.ndim(slm['coef']) == 3 and np.shape(slm['coef'])[2] == 1:
        x1, x2, x3 = np.shape(slm['coef'])
        slm['coef'] = slm['coef'].reshape(x1, x2)

    if np.ndim(slm['coef']) == 2:
        slm['k'] = np.array(1)
        slm['t'] = np.dot(h / (SSE2 + (SSE2<=0)) * (SSE2>0), df2/(df1-df2))
    elif np.ndim(slm['coef']) > 2:
        k2, v = np.shape(SSE2)
        k = np.around((np.sqrt(1 + 8*k2) -1)/2)
        slm['k'] = np.array(k)
        if k > 3:
            print('Roy''s max root for k>3 not programmed yet.')
            return

        l = min(k, df1-df2)
        slm['t'] = np.zeros((int(l),int(v)))

        if k == 2:
            det = SSE2[0,:] * SSE2[2,:] - SSE2[1,:]**2
            a11 = SSE2[2,:] * h[0,:] - SSE2[1,:] * h[1,:]
            a21 = SSE2[0,:] * h[1,:] - SSE2[1,:] * h[0,:]
            a12 = SSE2[2,:] * h[1,:] - SSE2[1,:] * h[2,:]
            a22 = SSE2[0,:] * h[2,:] - SSE2[1,:] * h[1,:]
            a0 = a11 * a22 - a12 * a21
            a1 = (a11 + a22) / 2
            s1 = np.array([sqrt(x) for x in  (a1**2-a0)]).real
            d = (df2 / (df1-df2)) / (det + (det<=0)) * (det>0)
            slm['t'][0,:] = (a1+s1) * d
            if l == 2:
                slm['t'][1,:] = (a1-s1) * d
        if k == 3:
            det = SSE2[0,:] * (SSE2[2,:] * SSE2[5,:] - SSE2[4,:]**2) - \
                  SSE2[5,:] * SSE2[1,:]**2 + \
                  SSE2[3,:] * (SSE2[1,:] * SSE2[4,:] * 2 - SSE2[2,:] * SSE2[3,:])
            m1 = SSE2[2,:] * SSE2[5,:] - SSE2[4,:]**2
            m3 = SSE2[0,:] * SSE2[5,:] - SSE2[3,:]**2
            m6 = SSE2[0,:] * SSE2[2,:] - SSE2[1,:]**2
            m2 = SSE2[3,:] * SSE2[4,:] - SSE2[1,:] * SSE2[5,:]
            m4 = SSE2[1,:] * SSE2[4,:] - SSE2[2,:] * SSE2[3,:]
            m5 = SSE2[1,:] * SSE2[3,:] - SSE2[0,:] * SSE2[4,:]
            a11 = m1 * h[0,:] + m2 * h[1,:] + m4 * h[3,:]
            a12 = m1 * h[1,:] + m2 * h[2,:] + m4 * h[4,:]
            a13 = m1 * h[3,:] + m2 * h[4,:] + m4 * h[5,:]
            a21 = m2 * h[0,:] + m3 * h[1,:] + m5 * h[3,:]
            a22 = m2 * h[1,:] + m3 * h[2,:] + m5 * h[4,:]
            a23 = m2 * h[3,:] + m3 * h[4,:] + m5 * h[5,:]
            a31 = m4 * h[0,:] + m5 * h[1,:] + m6 * h[3,:]
            a32 = m4 * h[1,:] + m5 * h[2,:] + m6 * h[4,:]
            a33 = m4 * h[3,:] + m5 * h[4,:] + m6 * h[5,:]
            a0 = -a11 * (a22*a33 - a23*a32) + a12 * (a21*a33 - a23*a31) - \
                 a13 * (a21*a32 - a22*a31)
            a1 = a22*a33 - a23*a32 + a11*a33 - a13*a31 + a11*a22 - a12*a21
            a2 = -(a11 + a22 + a33)
            q = a1/3-a2**2/9
            r = (a1*a2 - 3*a0)/6 - a2**3/27
            s1 = (r + [sqrt(x) for x in  (q**3 + r**2)])**(1/3)
            z = np.zeros((3,v))
            z[0,:] = 2 * s1.real - a2/3
            z[1,:] = -s1.real - a2/3 + np.sqrt(3) * s1.imag
            z[2,:] = -s1.real - a2/3 - np.sqrt(3) * s1.imag

            if  not np.count_nonzero(z) == 0:
                z.sort(axis=0)
                z = z[::-1]
            d = (df2/(df1-df2) / (det + (det<=0)) * (det>0) )

            for j in range(0, l):
                slm['t'][j,:] = z[j,:] * d
    return slm


def linear_model(Y, M, surf=None, niter=1, thetalim=0.01, drlim=0.1):
    """ Fits linear mixed effects models to surface data and estimates resels.

    Parameters
    ----------
    Y : ndarray, shape = (n_samples, n_verts) or (n_samples, n_verts, n_feats)
        Surface data.
    M : Term or Random
        Design matrix.
    surf : dict or BSPolyData, optional
        Surface triangles (surf['tri']) or volumetric data (surf['lat']).
        If 'tri', shape = (n_edges, 2). If 'lat', then it is a boolean 3D
        array. Alternatively a BSPolyData object can be provided. Default is None.
    niter : int, optional
        Number of extra iterations of the Fisher scoring algorithm for fitting
        mixed effects models. Default is 1.
    thetalim : float, optional
        Lower limit on variance coefficients, in sd's. Default is 0.01.
    drlim : float, optional
        Step of ratio of variance coefficients, in sd's. Default 0.1.

    Returns
    -------
    slm : dict
        Dictionary with the following keys:

        - 'X' : ndarray, shape = (n_samples, n_pred)
            Design matrix.
        - 'df' : int
            Degrees of freedom.
        - 'coef' : ndarray, shape = (n_pred, n_verts)
            Model coefficients.
        - 'SSE' : ndarray, shape = (n_feat, n_verts)
            Sum of square errors.
        - 'V' : ndarray, shape = (n_samples, n_samples, n_rand)
            Variance matrix bases. Only when mixed effects.
        - 'r' : ndarray, shape = (n_rand - 1, n_verts)
            Coefficients of the first (q-1) components of 'V' divided by their
            sum. Coefficients are clamped to a minimum of 0.01 x sd.
            Only when mixed effects.
        - 'dr' : ndarray
             Vector of increments in 'r' = 0.1 x sd
        - 'resl' : ndarray, (n_edges, n_feat)
            Sum over observations of squares of differences of normalized
            residuals along each edge. Only when ``surf is not None``.
        - 'tri' : ndarray, (n_cells, 3)
            Cells in surf. Only when ``surf is not None``.
        - 'lat' : ndarray
            Neighbors in lattice.

    """

    n, v = Y.shape[:2]  # number of samples x number of points
    k = 1 if Y.ndim == 2 else Y.shape[2]  # number of features

    # Get data from term/random
    V = None
    if isinstance(M, Random):
        X, Vl = M.mean.matrix.values, M.variance.matrix.values

        # check in var contains intercept (constant term)
        n2, q = Vl.shape
        II = np.identity(n).ravel()

        r = II - Vl @ (la.pinv(Vl) @ II)
        if (r ** 2).mean() > np.finfo(float).eps:
            warnings.warn('Did you forget an error term, I? :-)')

        if q > 1 or q == 1 and np.abs(II - Vl.T).sum() > 0:
            V = Vl.reshape(n, n, -1)

    else:  # No random term
        q = 1
        if isinstance(M, Term):
            X = M.matrix.values
        else:
            if M.size > 1:
                warnings.warn('If you don''t convert vectors to terms you can '
                              'get unexpected results :-(')
            X = M

        if X.shape[0] == 1:
            X = np.tile(X, (n, 1))

    # check if term (x) contains intercept (constant term)
    pinvX = la.pinv(X)
    r = 1 - X @ pinvX.sum(1)
    if (r ** 2).mean() > np.finfo(float).eps:
        warnings.warn('Did you forget an error term, I? :-)')

    p = X.shape[1]  # number of predictors
    df = n - la.matrix_rank(X)  # degrees of freedom

    slm = dict(df=df, X=X)

    if k == 1:  # Univariate

        if q == 1:  # Fixed effects

            if V is None:  # OLS
                coef = pinvX @ Y
                Y = Y - X @ coef

            else:
                V = V / np.diag(V).mean(0)
                Vmh = la.inv(la.cholesky(V).T)

                coef = (la.pinv(Vmh @ X) @ Vmh) @ Y
                Y = Vmh @ Y - (Vmh @ X) @ coef

            sse = np.sum(Y ** 2, axis=0)

        else:  # mixed effects

            q1 = q - 1

            V /= np.diagonal(V, axis1=0, axis2=1).mean(-1)
            slm_r = np.zeros((q1, v))

            # start Fisher scoring algorithm
            R = np.eye(n) - X @ la.pinv(X)
            RVV = (V.T @ R.T).T
            E = (Y.T @ (R.T @ RVV.T))
            E *= Y.T
            E = E.sum(-1)

            RVV2 = np.zeros([n, n, q])
            E2 = np.zeros([q, v])
            for j in range(q):
                RV2 = R @ V[..., j]
                E2[j] = (Y * ((RV2 @ R) @ Y)).sum(0)
                RVV2[..., j] = RV2

            M = np.einsum('ijk,jil->kl', RVV, RVV, optimize='optimal')

            theta = la.pinv(M) @ E
            tlim = np.sqrt(2*np.diag(la.pinv(M))) * thetalim
            tlim = tlim[:, None] * theta.sum(0)
            m = theta < tlim
            theta[m] = tlim[m]
            r = theta[:q1] / theta.sum(0)

            Vt = 2*la.pinv(M)
            m1 = np.diag(Vt)
            m2 = 2 * Vt.sum(0)
            Vr = m1[:q1]-m2[:q1] * slm_r.mean(1) + Vt.sum()*(r**2).mean(-1)
            dr = np.sqrt(Vr) * drlim

            # Extra Fisher scoring iterations
            for it in range(niter):
                irs = np.round(r.T / dr)
                ur, jr = np.unique(irs, axis=0, return_inverse=True)
                nr = ur.shape[0]
                for ir in range(nr):
                    iv = jr == ir
                    rv = r[:, iv].mean(1)

                    Vs = (1-rv.sum()) * V[..., q-1]
                    Vs += (V[..., :q1] * rv).sum(-1)

                    Vinv = la.inv(Vs)
                    VinvX = Vinv @ X
                    G = la.pinv(X.T @ VinvX) @ VinvX.T
                    R = Vinv - VinvX @ G

                    RVV = (V.T @ R.T).T
                    E = (Y[:, iv].T @ (R.T @ RVV.T))
                    E *= Y[:, iv].T
                    E = E.sum(-1)

                    M = np.einsum('ijk,jil->kl', RVV, RVV, optimize='optimal')

                    thetav = la.pinv(M) @ E
                    tlim = np.sqrt(2*np.diag(la.pinv(M))) * thetalim
                    tlim = tlim[:, None] * thetav.sum(0)

                    m = thetav < tlim
                    thetav[m] = tlim[m]
                    theta[:, iv] = thetav

                r = theta[:q1] / theta.sum(0)

            # finish Fisher scoring
            irs = np.round(r.T / dr)
            ur, jr = np.unique(irs, axis=0, return_inverse=True)
            nr = ur.shape[0]

            coef = np.zeros((p, v))
            sse = np.zeros(v)
            for ir in range(nr):
                iv = jr == ir
                rv = r[:, iv].mean(1)

                Vs = (1 - rv.sum()) * V[..., q - 1]
                Vs += (V[..., :q1] * rv).sum(-1)

                # Vmh = la.inv(la.cholesky(Vs).T)
                Vmh = la.inv(la.cholesky(Vs))
                VmhX = Vmh @ X
                G = (la.pinv(VmhX.T @ VmhX) @ VmhX.T) @ Vmh

                coef[:, iv] = G @ Y[:, iv]
                R = Vmh - VmhX @ G
                Y[:, iv] = R @ Y[:, iv]
                sse[iv] = (Y[:, iv]**2).sum(0)

            slm.update(dict(r=r, dr=dr[:, None]))

        sse = sse[None]

    else:  # multivariate
        if q > 1:
            raise ValueError('Multivariate mixed effects models not yet '
                             'implemented :-(')

        if V is None:
            X2 = X
        else:
            V = V / np.diag(V).mean(0)
            Vmh = la.inv(la.cholesky(V)).T
            X2 = Vmh @ X
            pinvX = la.pinv(X2)
            Y = Vmh @ Y

        coef = pinvX @ Y.T.swapaxes(-1, -2)
        Y = Y - (X2 @ coef).swapaxes(-1, -2).T
        coef = coef.swapaxes(-1, -2).T

        k2 = k * (k + 1) // 2
        sse = np.zeros((k2, v))
        j = -1
        for j1 in range(k):
            for j2 in range(j1+1):
                j = j + 1
                sse[j] = (Y[..., j1]*Y[..., j2]).sum(0)

    slm.update(dict(coef=coef, SSE=sse))
    if V is not None:
        slm['V'] = V

    if surf is not None and (isinstance(surf,BSPolyData) or ('tri' in surf or 'lat' in surf)):
        if isinstance(surf,BSPolyData):
            slm['tri'] = np.array(get_cells(surf)) + 1
        else:
            key = 'tri' if 'tri' in surf else 'lat'
            slm[key] = surf[key]

        edges = mesh_edges(surf)

        n_edges = edges.shape[0]

        resl = np.zeros((n_edges, k))
        Y = np.atleast_3d(Y)

        for j in range(k):
            normr = np.sqrt(sse[((j+1) * (j+2) // 2) - 1])
            for i in range(n):
                u = Y[i, :, j] / normr
                resl[:, j] += np.diff(u[edges], axis=1).ravel()**2
        slm['resl'] = resl

    return slm


def _t_test(slm, contrast):

    # T statistics for a contrast in a univariate or multivariate model.
    # Inputs
    # slm         = a dict with mandatory keys 'X', 'df', 'coef', 'SSE'
    # slm['X']    = numpy array of shape (n x p), design matrix.
    # slm['df']   = int, float or numpy array of shape (a,), degrees of freedom
    # slm['coef'] = numpy array of shape (p x v) or (p x v x k)
    #             = array of coefficients of the linear model.
    #             = if (p x v), then k is assigned to 1 here.
    # slm['SSE']  = numpy array of shape (k*(k+1)/2 x v)
    #             = array of sum of squares of errors
    # slm['V']    = numpy array of shape (n x n x q), variance array bases,
    #             = normalised so that mean(diag(slm['V']))=1. If absent, assumes q=1
    #             and slm.V=eye(n).
    # slm['r']    = numpy array of shape ((q-1) x v), coefficients of the
    #             first (q-1) components of slm['V'] divided by their sum.
    # slm['dr']   = numpy array of shape ((q-1), ), increments in slm['r']
    # contrast    = numpy array of shape (n x 1), contrasts in the observations, ie.,
    #             = slm['X']*slm['c'].T, where slm['c'] is a contrast in slm.coef, or,
    #             = numpy array of shape (1 x p), of slm.c,
    #             padded with 0's if len(contrast)<p.
    # Outputs
    # slm['c']    = numpy array of shape (1 x p), contrasts in coefficents of the
    #             linear model.
    # slm['k']    = number of variates
    # slm['ef']   = numpy array of shape (k x v), array of effects.
    # slm['sd']   = numpy array of shape (k x v), standard deviations of the effects.
    # slm['t']    = numpy array of shape (1 x v), array of T = ef/sd if k=1, or,
    #             Hotelling's T if k=2 or 3, defined as the maximum T over all linear
    #             combinations of the k variates, k>3 is not programmed yet.
    # slm['dfs']  = numpy array of shape (1 x v), effective degrees of freedom.
    #             Absent if q=1.

    #% Note that the contrast in the observations is used to determine the
    #% intended contrast in the model coefficients, slm.c. However there is some
    #% ambiguity in this when the model contains redundant terms. An example of
    #% such a model is 1 + Gender (Gender by itself does not contain redundant
    #% terms). Only one of the ambiguous contrasts is estimable (i.e. has slm.sd
    #% < Inf), and this is the one chosen, though it may not be the contrast
    #% that you intended. To check this, compare the contrast in the
    #% coefficients slm.c to the actual design matrix in slm.X. Note that the
    #% redundant columns of the design matrix have weights given by the rows of
    #% null(slm.X,'r')'

    def null(A, eps=1e-15):
        u, s, vh = scipy.linalg.svd(A)
        null_mask = (s <= eps)
        null_space = scipy.compress(null_mask, vh, axis=0)
        return scipy.transpose(null_space)

    if not isinstance(slm['df'], np.ndarray):
        slm['df'] =np.array([slm['df']])

    if contrast.ndim == 1:
        contrast = np.reshape(contrast, (-1, 1))

    [n, p] = np.shape(slm['X'])
    pinvX  = np.linalg.pinv(slm['X'])

    if len(contrast) <= p:
        c = np.concatenate((contrast, \
                            np.zeros((1, p-np.shape(contrast)[1]))), axis=1).T

        if np.square(np.dot(null_space(slm['X']).T, c)).sum()  \
                / np.square(c).sum() > np.spacing(1):
            sys.exit('Contrast is not estimable :-(')

    else:
        c = np.dot(pinvX, contrast)
        r = contrast - np.dot(slm['X'], c)

        if np.square(np.ravel(r, 'F')).sum() \
                / np.square(np.ravel(contrast, 'F')).sum() >  np.spacing(1) :
            warnings.warn('Contrast is not in the model :-( ')

    slm['c']  = c.T
    slm['df'] = slm['df'][len(slm['df'])-1 ]

    if np.ndim(slm['coef']) == 2:
        k = 1
        slm['k'] = k

        if not 'r' in slm.keys():
            # fixed effect
            if 'V' in slm.keys():
                Vmh = np.linalg.inv(cholesky(slm['V']).T)
                pinvX = np.linalg.pinv(np.dot(Vmh, slm['X']))

            Vc = np.sum(np.square(np.dot(c.T, pinvX)), axis=1)
        else:
            # mixed effect
            q1, v = np.shape(slm['r'])
            q = q1 + 1
            nc = np.shape(slm['dr'])[1]
            chunck = np.ceil(v / nc)
            irs = np.zeros((q1, v))

            for ic in range(1, nc+1):
                v1 = 1 + (ic - 1) * chunck
                v2 = np.min((v1 + chunck - 1, v))
                vc = v2 - v1 + 1

                irs[:, int(v1-1):int(v2)] = np.around(np.multiply(\
                     slm['r'][:, int(v1-1):int(v2)], \
                     np.tile(1/slm['dr'][:,(ic-1)], (1,vc))))

            ur, ir, jr = np.unique(irs, axis=0, return_index=True, return_inverse=True)
            ir = ir + 1
            jr = jr + 1
            nr = np.shape(ur)[0]
            slm['dfs'] = np.zeros((1,v))
            Vc = np.zeros((1,v))


            for ir in range(1, nr+1):
                iv = (jr == ir).astype(int)
                rv = slm['r'][:, (iv-1)].mean(axis=1)
                V = (1 - rv.sum()) * slm['V'][:,:,(q-1)]

                for j in range(1, q1+1):
                    V = V + rv[(j-1)] * slm['V'][:,:,(j-1)]

                Vinv = np.linalg.inv(V)
                VinvX = np.dot(Vinv, slm['X'])
                Vbeta = np.linalg.pinv(np.dot(slm['X'].T, VinvX))
                G = np.dot(Vbeta, VinvX.T)
                Gc = np.dot(G.T, c)
                R = Vinv - np.dot(VinvX, G)
                E = np.zeros((q,1))
                RVV = np.zeros((np.shape(slm['V'])))
                M = np.zeros((q,q))

                for j in range(1, q+1):
                    E[(j-1)] = np.dot(Gc.T, np.dot(slm['V'][:,:,(j-1)], Gc))
                    RVV[:,:,(j-1)] = np.dot(R, slm['V'][:,:,(j-1)])

                for j1 in range(1, q+1):
                    for j2 in range(j1, q+1):
                        M[(j1-1),(j2-1)] = (RVV[:,:,(j1-1)] * RVV[:,:,(j2-1)].T).sum()
                        M[(j2-1),(j1-1)] = M[(j1-1),(j2-1)]

                vc = np.dot(c.T, np.dot(Vbeta, c))
                iv = (jr == ir).astype(int)
                Vc[iv-1] = vc
                slm['dfs'][iv-1] = np.square(vc) / np.dot(E.T, \
                 np.dot(np.linalg.pinv(M), E))

        slm['ef'] = np.dot(c.T, slm['coef'])
        slm['sd'] = np.sqrt(np.multiply(Vc, slm['SSE']) / slm['df'])
        slm['t']  = np.multiply(np.divide(slm['ef'], (slm['sd']+(slm['sd']<= 0))), \
                                slm['sd']>0)

    else:
        # multivariate
        p, v, k   = np.shape(slm['coef'])
        slm['k']  = k
        slm['ef'] = np.zeros((k,v))

        for j in range(0,k):
            slm['ef'][j,:] = np.dot(c.T, slm['coef'][:,:,j])

        j  = np.arange(1, k+1)
        jj = (np.multiply(j, j+1)/2) - 1
        jj = jj.astype(int)

        vf =  np.divide(np.sum(np.square(np.dot(c.T, pinvX)), axis=1), slm['df'])
        slm['sd'] = np.sqrt(vf * slm['SSE'][jj,:])


        if k == 2:
            det = np.multiply(slm['SSE'][0,:], slm['SSE'][2,:]) - \
              np.square(slm['SSE'][1,:])

            slm['t'] = np.multiply(np.square(slm['ef'][0,:]), slm['SSE'][2,:]) + \
                   np.multiply(np.square(slm['ef'][1,:]), slm['SSE'][0,:]) - \
                   np.multiply(np.multiply(2 * slm['ef'][0,:], slm['ef'][1,:]), \
                   slm['SSE'][1,:])

        if k == 3:
            det = np.multiply(slm['SSE'][0,:], (np.multiply(slm['SSE'][2,:], \
              slm['SSE'][5,:]) - np.square(slm['SSE'][4,:]))) - \
              np.multiply(slm['SSE'][5,:], np.square(slm['SSE'][1,:])) + \
              np.multiply(slm['SSE'][3,:], (np.multiply(slm['SSE'][1,:], \
              slm['SSE'][4,:]) * 2 - np.multiply(slm['SSE'][2,:], slm['SSE'][3,:])))

            slm['t'] =  np.multiply(np.square(slm['ef'][0,:]), \
                    (np.multiply(slm['SSE'][2,:], slm['SSE'][5,:]) - \
                    np.square(slm['SSE'][4,:])))

            slm['t'] = slm['t'] + np.multiply(np.square(slm['ef'][1,:]), \
                    (np.multiply(slm['SSE'][0,:], slm['SSE'][5,:]) - \
                    np.square(slm['SSE'][3,:])))

            slm['t'] = slm['t'] + np.multiply(np.square(slm['ef'][2,:]), \
                    (np.multiply(slm['SSE'][0,:], slm['SSE'][2,:]) - \
                    np.square(slm['SSE'][1,:])))

            slm['t'] = slm['t'] + np.multiply(2*slm['ef'][0,:], \
                   np.multiply(slm['ef'][1,:], (np.multiply(slm['SSE'][3,:], \
                   slm['SSE'][4,:]) - np.multiply(slm['SSE'][1,:], slm['SSE'][5,:]))))

            slm['t'] = slm['t'] + np.multiply(2*slm['ef'][0,:], \
                   np.multiply(slm['ef'][2,:], (np.multiply(slm['SSE'][1,:], \
                   slm['SSE'][4,:]) - np.multiply(slm['SSE'][2,:], slm['SSE'][3,:]))))

            slm['t'] = slm['t'] + np.multiply(2*slm['ef'][1,:], \
                   np.multiply(slm['ef'][2,:], (np.multiply(slm['SSE'][1,:], \
                   slm['SSE'][3,:]) - np.multiply(slm['SSE'][0,:], slm['SSE'][4,:]))))

        if k > 3:
            sys.exit('Hotelling''s T for k>3 not programmed yet')

        slm['t'] = np.multiply(np.divide(slm['t'], (det + (det <= 0))), (det > 0)) / vf
        slm['t'] = np.multiply(np.sqrt(slm['t'] + (slm['t'] <= 0)), (slm['t'] > 0))

    return slm


def _peak_clus(slm, mask, thresh, reselspvert=None, edg=None):
    """ Finds peaks (local maxima) and clusters for surface data.
    Parameters
    ----------
    slm : a dictionary, mandatory keys: 't', 'tri' (or 'lat'),
        optional keys 'df', 'k'.
        slm['t'] : numpy array of shape (l,v),
            v is the number of vertices, the first row slm['t'][0,:] is used
            for the clusters, and the other rows are used to calculate cluster
            resels if slm['k']>1. See F for the precise definition
            of the extra rows.
        slm['tri'] : numpy array of shape (t,3), dype=int,
            triangle indices, values should be 1 and v,
        or,
        slm['lat'] : numpy array of shape (nx,nx,nz),
            values should be either 0 or 1.
            note that [nx,ny,nz]=size(volume).
        mask : numpy array of shape (v), dytpe=int,
            values should be either 0 or 1.
        thresh : float,
            clusters are vertices where slm['t'][0,mask]>=thresh.
        reselspvert : numpy array of shape (v),
            resels per vertex, by default: np.ones(v).
        edg :  numpy array of shape (e,2), dtype=int,
            edge indices, by default computed from mesh_edges function.
        slm['df'] : int,
            degrees of freedom, note that only the length (1 or 2) is used
            to determine if slm['t'] is Hotelling's T or T^2 when k>1.
        slm['k'] : int,
             k is number of variates, by default 1.

    Returns
    -------
    peak : a dictionary with keys 't', 'vertid', 'clusid'.
        peak['t'] : numpy array of shape (np,1),
            array of peaks (local maxima).
        peak['vertid] : numpy array of shape (np,1),
            array of vertex id's (1-based).
        peak['clusid'] : numpy array of shape (np,1),
            array of cluster id's that contain the peak.
    clus : a dictionary with keys 'clusid', 'nverts', 'resels'.
        clus['clusid'] : numpy array of shape (nc,1),
            array of cluster id numbers.
        clus['nverts'] : numpy array of shape (nc,1),
            array of number of vertices in the cluster.
        clus['resels'] : numpy array of shape (nc,1),
            array of resels in the cluster.
    clusid : numpy array of shape (1,v),
        array of cluster id's for each vertex.
    """
    if edg is None:
        edg = mesh_edges(slm)

    l, v = np.shape(slm['t'])
    slm_t = copy.deepcopy(slm['t'])
    slm_t[0, ~mask.astype(bool)] = slm_t[0,:].min()
    t1 = slm_t[0, edg[:,0]]
    t2 = slm_t[0, edg[:,1]]
    islm = np.ones((1,v))
    islm[0, edg[t1 < t2, 0]] = 0
    islm[0, edg[t2 < t1, 1]] = 0
    lmvox = np.argwhere(islm)[:,1] + 1
    excurset = np.array(slm_t[0,:] >= thresh, dtype=int)
    n = excurset.sum()

    if n < 1:
        peak = []
        clus = []
        clusid = []
        return peak, clus, clusid

    voxid = np.cumsum(excurset)
    edg = voxid[edg[np.all(excurset[edg],1), :]]
    nf = np.arange(1,n+1)

    # Find cluster id's in nf (from Numerical Recipes in C, page 346):
    for el in range(1, edg.shape[0]+1):
        j = edg[el-1, 0]
        k = edg[el-1, 1]
        while nf[j-1] != j:
            j = nf[j-1]
        while nf[k-1] != k:
            k = nf[k-1]
        if j != k:
            nf[j-1] = k

    for j in range(1, n+1):
        while nf[j-1] != nf[nf[j-1]-1]:
            nf[j-1] =  nf[nf[j-1]-1]

    vox = np.argwhere(excurset) + 1
    ivox = np.argwhere(np.in1d(vox, lmvox)) + 1
    clmid = nf[ivox-1]
    uclmid, iclmid, jclmid = np.unique(clmid,
                                       return_index=True, return_inverse=True)
    iclmid = iclmid +1
    jclmid = jclmid +1
    ucid = np.unique(nf)
    nclus = len(ucid)
    # implementing matlab's histc function ###
    bin_edges   = np.r_[-np.Inf, 0.5 * (ucid[:-1] + ucid[1:]), np.Inf]
    ucvol, ucvol_edges = np.histogram(nf, bin_edges)

    if reselspvert is None:
        reselsvox = np.ones(np.shape(vox))
    else:
        reselsvox = reselspvert[vox-1]

    # calling matlab-python version for scipy's interp1d
    nf1 = interp1(np.append(0, ucid), np.arange(0,nclus+1), nf,
                      kind='nearest')

    # if k>1, find volume of cluster in added sphere
    if 'k' not in slm or slm['k'] == 1:
        ucrsl = np.bincount(nf1.astype(int), reselsvox.flatten())
    if 'k' in slm and slm['k'] == 2:
        if l == 1:
            ndf = len(np.array([slm['df']]))
            r = 2 * np.arccos((thresh / slm_t[0, vox-1])**(float(1)/ndf))
        else:
            r = 2 * np.arccos(np.sqrt((thresh - slm_t[1,vox-1]) *
                                      (thresh >= slm_t[1,vox-1]) /
                                      (slm_t[0,vox-1] - slm_t[1,vox-1])))
        ucrsl =  np.bincount(nf1.astype(int), (r.T * reselsvox.T).flatten())
    if 'k' in slm and slm['k'] == 3:
        if l == 1:
            ndf = len(np.array([slm['df']]))
            r = 2 * math.pi * (1 - (thresh / slm_t[0, vox-1])**
                                (float(1)/ndf))
        else:
            nt = 20
            theta = (np.arange(1,nt+1,1) - 1/2) / nt * math.pi / 2
            s = (np.cos(theta)**2 * slm_t[1, vox-1]).T
            if l == 3:
                s =  s + ((np.sin(theta)**2) * slm_t[2,vox-1]).T
            r = 2 * math.pi * (1 - np.sqrt((thresh-s)*(thresh>=s) /
                                           (np.ones((nt,1)) *
                                            slm_t[0, vox-1].T -
                                            s ))).mean(axis=0)
        ucrsl = np.bincount(nf1.astype(int), (r.T * reselsvox.T).flatten())

    # and their ranks (in ascending order)
    iucrls = sorted(range(len(ucrsl[1:])), key=lambda k: ucrsl[1:][k])
    rankrsl = np.zeros((1, nclus))
    rankrsl[0, iucrls] =  np.arange(nclus,0,-1)

    lmid = lmvox[ismember(lmvox, vox)[0]]

    varA = slm_t[0, (lmid-1)]
    varB = lmid
    varC = rankrsl[0,jclmid-1]
    varALL = np.concatenate((varA.reshape(len(varA),1),
                             varB.reshape(len(varB),1),
                             varC.reshape(len(varC),1)), axis=1)
    lm = np.flipud(varALL[varALL[:,0].argsort(),])
    varNEW = np.concatenate((rankrsl.T, ucvol.reshape(len(ucvol),1),
                             ucrsl.reshape(len(ucrsl),1)[1:]) , axis=1)
    cl = varNEW[varNEW[:,0].argsort(),]
    clusid = np.zeros((1,v))
    clusid[0,(vox-1).T] = interp1(np.append(0, ucid),
                                      np.append(0, rankrsl), nf,
                                      kind='nearest')
    peak = {}
    peak['t'] = lm[:,0].reshape(len(lm[:,0]), 1)
    peak['vertid'] = lm[:,1].reshape(len(lm[:,1]), 1)
    peak['clusid'] = lm[:,2].reshape(len(lm[:,2]), 1)
    clus = {}
    clus['clusid'] = cl[:,0].reshape(len(cl[:,0]), 1)
    clus['nverts'] = cl[:,1].reshape(len(cl[:,1]), 1)
    clus['resels'] = cl[:,2] .reshape(len(cl[:,2]), 1)

    return peak, clus, clusid


def _resels(slm, mask=None):
    """ SurfStatResels of surface or volume data inside a mask.

    Parameters
    ----------
    slm : a dictionary with keys 'lat' or 'tri' and, optionally, 'resl'.
        slm['lat'] : a 3D numpy array of 1's and 0's.
        slm['tri'] : a 2D numpy array of shape (t, 3).
            Contains triangles of a surface. slm['tri'].max() is equal to the
            number of vertices.
        slm['resl'] : a 2D numpy array of shape (e, k).
            Sum over observations of squares of differences of normalized
            residuals along each edge.
    mask : a 1D numpy array of shape (v), dtype 'bool'.
        v must be equal to int(slm['tri'].max()).
        Contains 1's and 0's (1's are included and 0's are excluded).

    Returns
    -------
    resels : a 2D numpy array of shape (1, (D+1)).
        Array of 0,...,D dimensional resels of the mask, EC of the mask
        if slm['resl'] is not given.
    reselspvert : a 1D numpy array of shape (v).
        Array of D-dimensional resels per mask vertex.
    edg : a 2D numpy array of shape (e, 2).
        Array of edge indices.
    """

    def pacos(x):
        return np.arccos( np.minimum(np.abs(x),1) * np.sign(x) )

    if 'tri' in slm:
        # Get unique edges. Subtract 1 from edges to conform to Python's
        # counting from 0 - RV
        tri = np.sort(slm['tri'])-1
        edg = np.unique(np.vstack((tri[:,(0,1)], tri[:,(0,2)],
                                       tri[:,(1,2)])),axis=0)


        # If no mask is provided, create one with all included vertices set to
        # 1. If mask is provided, simply grab the number of vertices from mask.
        if mask is None:
            v = np.amax(edg)+1
            mask = np.full(v,False)
            mask[edg-1] = True
        else:
            #if np.ndim(mask) > 1:
            #    mask = np.squeeze(mask)
            #    if mask.shape[0] > 1:
            #        mask = mask.T
            v = mask.size

        ## Compute the Lipschitz–Killing curvatures (LKC)
        m = np.sum(mask)
        if 'resl' in slm:
            lkc = np.zeros((3,3))
        else:
            lkc = np.zeros((1,3))
        lkc[0,0] = m

        # LKC of edges
        maskedg = np.all(mask[edg],axis=1)
        lkc[0,1] = np.sum(maskedg)

        if 'resl' in slm:
            r1 = np.mean(np.sqrt(slm['resl'][maskedg,:]),axis=1)
            lkc[1,1] = np.sum(r1)
        # LKC of triangles
        # Made an adjustment from the MATLAB implementation:
        # The reselspvert computation is included in the if-statement.
        # MATLAB errors when the if statement is false as variable r2 is not
        # defined during the computation of reselspvert. - RV
        masktri = np.all(mask[tri],1)
        lkc[0,2] = np.sum(masktri)
        if 'resl' in slm:
            loc = row_ismember(tri[masktri,:][:,[0,1]], edg)
            l12 = slm['resl'][loc,:]
            loc = row_ismember(tri[masktri,:][:,[0,2]], edg)
            l13 = slm['resl'][loc,:]
            loc = row_ismember(tri[masktri,:][:,[1,2]], edg)
            l23 = slm['resl'][loc,:]
            a = np.fmax(4*l12*l13-(l12+l13-l23)**2,0)
            r2 = np.mean(np.sqrt(a),axis=1)/4
            lkc[1,2] = np.sum(np.mean(np.sqrt(l12) +
               np.sqrt(l13)+np.sqrt(l23),axis=1))/2
            lkc[2,2] = np.nansum(r2,axis=0)

            # Compute resels per mask vertex
            reselspvert = np.zeros(v)
            for j in range(0,3):
                reselspvert = reselspvert + \
                        np.bincount(tri[masktri,j], weights=r2, minlength=v)
            D = 2
            reselspvert = reselspvert / (D+1) / np.sqrt(4*np.log(2)) ** D
        else:
            reselspvert = None

    if 'lat' in slm:
        edg = mesh_edges(slm)
        # The lattice is filled with 5 alternating tetrahedra per cube
        I, J, K = np.shape(slm['lat'])
        IJ = I*J
        i, j = np.meshgrid(range(1,I+1),range(1,J+1))
        i = np.squeeze(np.reshape(i,(-1,1)))
        j = np.squeeze(np.reshape(j,(-1,1)))

        c1  = np.argwhere((((i+j)%2)==0) & (i < I) & (j < J))
        c2  = np.argwhere((((i+j)%2)==0) & (i > 1) & (j < J))
        c11 = np.argwhere((((i+j)%2)==0) & (i == I) & (j < J))
        c21 = np.argwhere((((i+j)%2)==0) & (i == I) & (j > 1))
        c12 = np.argwhere((((i+j)%2)==0) & (i < I) & (j == J))
        c22 = np.argwhere((((i+j)%2)==0) & (i > 1) & (j == J))

        # outcome is 1 lower than MATLAB due to 0-1 counting difference. - RV
        d1  = np.argwhere((((i+j)%2)==1) & (i < I) & (j < J))+IJ
        d2  = np.argwhere((((i+j)%2)==1) & (i > 1) & (j < J))+IJ

        tri1 = cat((
            cat((c1, c1+1, c1+1+I),axis=1),
            cat((c1, c1+I, c1+1+I),axis=1),
            cat((c2-1, c2, c2-1+I),axis=1),
            cat((c2, c2-1+I, c2+I),axis=1)),
            axis=0)
        tri2= cat((
            cat((c1,    c1+1,    c1+1+IJ),axis=1),
            cat((c1,    c1+IJ,   c1+1+IJ),axis=1),
            cat((c1,    c1+I,    c1+I+IJ),axis=1),
            cat((c1,     c1+IJ,   c1+I+IJ),axis=1),
            cat((c1,     c1+1+I,  c1+1+IJ),axis=1),
            cat((c1,     c1+1+I,  c1+I+IJ),axis=1),
            cat((c1,     c1+1+IJ, c1+I+IJ),axis=1),
            cat((c1+1+I, c1+1+IJ, c1+I+IJ),axis=1),
            cat((c2-1,   c2,      c2-1+IJ),axis=1),
            cat((c2,     c2-1+IJ, c2+IJ),axis=1),
            cat((c2-1,   c2-1+I,  c2-1+IJ),axis=1),
            cat((c2-1+I, c2-1+IJ, c2-1+I+IJ),axis=1),
            cat((c2,     c2-1+I,  c2+I+IJ),axis=1),
            cat((c2,     c2-1+IJ, c2+I+IJ),axis=1),
            cat((c2,     c2-1+I,  c2-1+IJ),axis=1),
            cat((c2-1+I, c2-1+IJ, c2+I+IJ),axis=1),
            cat((c11,    c11+I,    c11+I+IJ),axis=1),
            cat((c11,    c11+IJ,   c11+I+IJ),axis=1),
            cat((c21-I,  c21,      c21-I+IJ),axis=1),
            cat((c21,    c21-I+IJ, c21+IJ),axis=1),
            cat((c12,    c12+1,    c12+1+IJ),axis=1),
            cat((c12,    c12+IJ,   c12+1+IJ),axis=1),
            cat((c22-1,  c22,      c22-1+IJ),axis=1),
            cat((c22,    c22-1+IJ, c22+IJ),axis=1)),
            axis=0)
        tri3 = cat((
            cat((d1,     d1+1,    d1+1+I),axis=1),
            cat((d1,     d1+I,    d1+1+I),axis=1),
            cat((d2-1,   d2,      d2-1+I),axis=1),
            cat((d2,     d2-1+I,  d2+I),axis=1)),
            axis=0)
        tet1 = cat((
            cat((c1,     c1+1,    c1+1+I,    c1+1+IJ),axis=1),
            cat((c1,     c1+I,    c1+1+I,    c1+I+IJ),axis=1),
            cat((c1,     c1+1+I,  c1+1+IJ,   c1+I+IJ),axis=1),
            cat((c1,     c1+IJ,   c1+1+IJ,   c1+I+IJ),axis=1),
            cat((c1+1+I, c1+1+IJ, c1+I+IJ,   c1+1+I+IJ),axis=1),
            cat((c2-1,   c2,      c2-1+I,    c2-1+IJ),axis=1),
            cat((c2,     c2-1+I,  c2+I,      c2+I+IJ),axis=1),
            cat((c2,     c2-1+I,  c2-1+IJ,   c2+I+IJ),axis=1),
            cat((c2,     c2-1+IJ, c2+IJ,     c2+I+IJ),axis=1),
            cat((c2-1+I, c2-1+IJ, c2-1+I+IJ, c2+I+IJ),axis=1)),
            axis=0)

        v = np.int(np.round(np.sum(slm['lat'])))
        if mask is None:
            mask = np.ones(v,dtype=bool)

        reselspvert = np.zeros(v)
        vs = np.cumsum(np.squeeze(np.sum(np.sum(slm['lat'],axis=0),axis=0)))
        vs = cat((np.zeros(1),vs,np.expand_dims(vs[K-1],axis=0)),axis=0)
        vs = vs.astype(int)
        es = 0
        lat = np.zeros((I,J,2))
        lat[:,:,0] = slm['lat'][:,:,0]
        lkc = np.zeros((4,4))
        for k in range(0,K):
            f = (k+1) % 2
            if k < (K-1):
                lat[:,:,f] = slm['lat'][:,:,k+1]
            else:
                lat[:,:,f] = np.zeros((I,J))
            vid = (np.cumsum(lat.flatten('F')) * np.reshape(lat.T,-1)).astype(int)
            if f:
                edg1 = edg[np.logical_and(edg[:,0]>(vs[k]-1), \
                                          edg[:,0] <= (vs[k+1]-1)),:]-vs[k]
                edg2 = edg[np.logical_and(edg[:,0] > (vs[k]-1), \
                                          edg[:,1] <= (vs[k+2]-1)),:]-vs[k]
                # Added a -1 - RV
                tri = cat((vid[tri1[np.all(np.reshape(lat.flatten('F')[tri1], \
                                                      tri1.shape),1),:]],
                           vid[tri2[np.all(np.reshape(lat.flatten('F')[tri2], \
                                                      tri2.shape),1),:]]),
                           axis=0)-1
                mask1 = mask[np.arange(vs[k],vs[k+2])]
            else:
                edg1 = cat((
                    edg[np.logical_and(edg[:,0]  > (vs[k]-1), edg[:,1] <= \
                        (vs[k+1]-1)), :] - vs[k] + vs[k+2] - vs[k+1],
                    cat((
                        np.expand_dims(edg[np.logical_and(edg[:,0] <=        \
                                                          (vs[k+1]-1),       \
                                                          edg[:,1] >         \
                                                          (vs[k+1]-1)), 1]   \
                                                          - vs[k+1],axis=1),
                        np.expand_dims(edg[np.logical_and(edg[:,0] <=        \
                                                          (vs[k+1]-1),       \
                                                          edg[:,1] >         \
                                                          (vs[k+1]-1)), 0]   \
                                                          - vs[k] + vs[k+2]  \
                                                          - vs[k+1],axis=1)),
                        axis=1)),
                    axis=0)
                edg2 = cat((edg1, edg[np.logical_and(edg[:,0] > (vs[k+1]-1), \
                           edg[:,1] <= (vs[k+2]-1)),:] - vs[k+1]), axis=0)
                # Added a -1 - RV
                tri = cat((vid[tri3[np.all(lat.flatten('F')[tri3],axis=1),:]],
                          vid[tri2[np.all(lat.flatten('F')[tri2],axis=1),:]]),
                          axis=0)-1
                mask1 = cat((
                    mask[np.arange(vs[k+1], vs[k+2])],
                    mask[np.arange(vs[k],   vs[k+1])]))
            # Added a -1 -RV
            tet = vid[tet1[np.all(lat.flatten('F')[tet1],axis=1),:]]-1
            m1 = np.max(edg2[:,0])
            ue = edg2[:,0] + m1 * (edg2[:,1]-1)
            e = edg2.shape[0]
            ae = np.arange(0,e)
            if e < 2 ** 31:
                sparsedg = csr_matrix((ae,(ue,np.zeros(ue.shape,dtype=int))),
                                      dtype=np.int)
                sparsedg.eliminate_zeros()
            ##
            lkc1 = np.zeros((4,4))
            lkc1[0,0] = np.sum(mask[np.arange(vs[k],vs[k+1])])

            ## LKC of edges
            maskedg = np.all(mask1[edg1],axis=1)

            lkc1[0,1] = np.sum(maskedg)
            if 'resl' in slm:
                r1 = np.mean(np.sqrt(slm['resl'][np.argwhere(maskedg)+es,:]),
                             axis=1)
                lkc1[1,1] = np.sum(r1)

            ## LKC of triangles
            masktri = np.all(mask1[tri],axis=1).flatten()
            lkc1[0,2] = np.sum(masktri)
            if 'resl' in slm:
                if all(masktri == False):
                    # Set these variables to empty arrays to match the MATLAB
                    # implementation.
                    lkc1[1,2] = 0
                    lkc1[2,2] = 0
                else:
                    if e < 2 ** 31:
                        l12 = slm['resl'][sparsedg[tri[masktri,0] + m1 * \
                                        (tri[masktri,1]-1), 0].toarray() + es, :]
                        l13 = slm['resl'][sparsedg[tri[masktri,0] + m1 * \
                                        (tri[masktri,2]-1), 0].toarray() + es, :]
                        l23 = slm['resl'][sparsedg[tri[masktri,1] + m1 * \
                                        (tri[masktri,2]-1), 0].toarray() + es, :]
                    else:
                        l12 = slm['resl'][interp1(ue,ae,tri[masktri,0] + m1 * \
                                (tri[masktri,1] - 1),kind='nearest') + es, :]
                        l13 = slm['resl'][interp1(ue,ae,tri[masktri,0] + m1 * \
                                (tri[masktri,2] - 1),kind='nearest') + es, :]
                        l23 = slm['resl'][interp1(ue,ae,tri[masktri,1] + m1 * \
                                (tri[masktri,2] - 1),kind='nearest') + es, :]
                    a = np.fmax(4 * l12 * l13 - (l12+l13-l23) ** 2, 0)
                    r2 = np.mean(np.sqrt(a),axis=1)/4
                    lkc1[1,2] = np.sum(np.mean(np.sqrt(l12) + np.sqrt(l13) +
                                    np.sqrt(l23),axis=1))/2
                    lkc1[2,2] = np.sum(r2)

                # The following if-statement has nargout >=2 in MATLAB,
                # but there's no Python equivalent so ignore that. - RV
                if K == 1:
                    for j in range(0,3):
                        if f:
                            v1 = tri[masktri,j] + vs[k]
                        else:
                            v1 = tri[masktri,j] + vs[k+1]
                            v1 = v1 - int(vs > vs[k+2]) * (vs[k+2]-vs[k])
                        reselspvert += np.bincount(v1, r2, v)

            ## LKC of tetrahedra
            masktet = np.all(mask1[tet],axis=1).flatten()
            lkc1[0,3] = np.sum(masktet)
            if 'resl' in slm and k < (K-1):
                if e < 2 ** 31:
                    l12 = slm['resl'][(sparsedg[tet[masktet,0] + m1 * \
                                  (tet[masktet,1]-1),0].toarray() + es).tolist(), :]
                    l13 = slm['resl'][(sparsedg[tet[masktet,0] + m1 * \
                                  (tet[masktet,2]-1),0].toarray() + es).tolist(), :]
                    l23 = slm['resl'][(sparsedg[tet[masktet,1] + m1 * \
                                  (tet[masktet,2]-1),0].toarray() + es).tolist(), :]
                    l14 = slm['resl'][(sparsedg[tet[masktet,0] + m1 * \
                                  (tet[masktet,3]-1),0].toarray() + es).tolist(), :]
                    l24 = slm['resl'][(sparsedg[tet[masktet,1] + m1 * \
                                  (tet[masktet,3]-1),0].toarray() + es).tolist(), :]
                    l34 = slm['resl'][(sparsedg[tet[masktet,2] + m1 * \
                                  (tet[masktet,3]-1),0].toarray() + es).tolist(), :]
                else:
                    l12 = slm['resl'][interp1(ue,ae,tet[masktet,0] + m1 * \
                              (tet[masktet,1]-1),kind='nearest')+es,:]
                    l13 = slm['resl'][interp1(ue,ae,tet[masktet,0] + m1 * \
                              (tet[masktet,2]-1),kind='nearest')+es,:]
                    l23 = slm['resl'][interp1(ue,ae,tet[masktet,1] + m1 * \
                              (tet[masktet,2]-1),kind='nearest')+es,:]
                    l14 = slm['resl'][interp1(ue,ae,tet[masktet,0] + m1 * \
                              (tet[masktet,3]-1),kind='nearest')+es,:]
                    l24 = slm['resl'][interp1(ue,ae,tet[masktet,1] + m1 * \
                              (tet[masktet,3]-1),kind='nearest')+es,:]
                    l34 = slm['resl'][interp1(ue,ae,tet[masktet,2] + m1 * \
                              (tet[masktet,3]-1),kind='nearest')+es,:]
                a4 = np.fmax(4 * l12 * l13 - (l12 + l13 -l23) ** 2, 0)
                a3 = np.fmax(4 * l12 * l14 - (l12 + l14 -l24) ** 2, 0)
                a2 = np.fmax(4 * l13 * l14 - (l13 + l14 -l34) ** 2, 0)
                a1 = np.fmax(4 * l23 * l24 - (l23 + l24 -l34) ** 2, 0)

                d12 = 4 * l12 * l34 - (l13 + l24 - l23 - l14) ** 2
                d13 = 4 * l13 * l24 - (l12 + l34 - l23 - l14) ** 2
                d14 = 4 * l14 * l23 - (l12 + l34 - l24 - l13) ** 2

                h = np.logical_or(a1 <= 0, a2 <= 0)
                delta12 = np.sum(np.mean(np.sqrt(l34) * pacos((d12-a1-a2) / \
                            np.sqrt(a1 * a2 + h) / 2 * (1-h) + h),axis=1))
                h = np.logical_or(a1 <= 0, a3 <= 0)
                delta13 = np.sum(np.mean(np.sqrt(l24) * pacos((d13-a1-a3) / \
                            np.sqrt(a1 * a3 + h) / 2 * (1-h) + h),axis=1))
                h = np.logical_or(a1 <= 0, a4 <= 0)
                delta14 = np.sum(np.mean(np.sqrt(l23) * pacos((d14-a1-a4) / \
                            np.sqrt(a1 * a4 + h) / 2 * (1-h) + h),axis=1))
                h = np.logical_or(a2 <= 0, a3 <= 0)
                delta23 = np.sum(np.mean(np.sqrt(l14) * pacos((d14-a2-a3) / \
                            np.sqrt(a2 * a3 + h) / 2 * (1-h) + h),axis=1))
                h = np.logical_or(a2 <= 0, a4 <= 0)
                delta24 = np.sum(np.mean(np.sqrt(l13) * pacos((d13-a2-a4) / \
                            np.sqrt(a2 * a4 + h) / 2 * (1-h) + h),axis=1))
                h = np.logical_or(a3 <= 0, a4 <= 0)
                delta34 = np.sum(np.mean(np.sqrt(l12) * pacos((d12-a3-a4) / \
                            np.sqrt(a3 * a4 + h) / 2 * (1-h) + h),axis=1))

                r3 = np.squeeze(np.mean(np.sqrt(np.fmax((4 * a1 * a2 - \
                                (a1 + a2 - d12) **2) / (l34 + (l34<=0)) * \
                                (l34>0), 0)),axis=1) / 48)

                lkc1[1,3] = (delta12 + delta13 + delta14 + delta23 + delta24 +
                            delta34)/(2 * np.pi)
                lkc1[2,3] = np.sum(np.mean(np.sqrt(a1) + np.sqrt(a2) +
                            np.sqrt(a3) + np.sqrt(a4), axis=1))/8
                lkc1[3,3] = np.sum(r3)

                ## Original MATLAB code has a if nargout>=2 here, ignore it
                # as no equivalent exists in Python - RV.
                for j in range(0,4):
                    if f:
                        v1 = tet[masktet,j] + vs[k]
                    else:
                        v1 = tet[masktet,j] + vs[k+1]
                        v1 = v1 - (v1 > (vs[k+2]-1)) * (vs[k+2] - vs[k])
                    if np.ndim(r3) == 0:
                        r3 = r3.tolist()
                        r3 = [r3]
                    reselspvert += np.bincount(v1, r3, v)
            lkc = lkc + lkc1
            es = es + edg1.shape[0]

        ## Original MATLAB code has a if nargout>=2 here,
        # ignore it as no equivalent exists in Python - RV.
        D = 2 + (K>1)
        reselspvert = reselspvert / (D+1) / np.sqrt(4*np.log(2)) ** D

    ## Compute resels - RV
    D1 = lkc.shape[0]-1
    D2 = lkc.shape[1]-1
    tpltz = toeplitz((-1)**(np.arange(0,D1+1)), (-1)**(np.arange(0,D2+1)))
    lkcs = np.sum(tpltz * lkc, axis=1).T
    lkcs = np.trim_zeros(lkcs,trim='b')
    lkcs = np.atleast_2d(lkcs)
    D = lkcs.shape[1]-1
    resels = lkcs / np.sqrt(4*np.log(2))**np.arange(0,D+1)

    return resels, reselspvert, edg
