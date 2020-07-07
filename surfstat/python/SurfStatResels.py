import numpy as np
from matlab_functions import ismember
from scipy.linalg import toeplitz
from SurfStatEdg import py_SurfStatEdg

def pacos(x):
    return np.arccos( np.minimum(np.abs(x),1) * np.sign(x) ) 

def py_SurfStatResels(slm, mask=None):
    if 'tri' in slm:
        # Get unique edges. Subtract 1 from edges to conform to Python's counting from 0 - RV
        tri = np.sort(slm['tri']) - 1
        edg = np.unique(np.vstack((tri[:,(0,1)], tri[:,(0,2)], tri[:,(1,2)])),axis=0)

        # If no mask is provided, create one with all included vertices set to 1. - RV
        # If one is provided, simply grab the number of vertices from the mask. - RV  
        if mask is None:
            v = np.amax(edg)+1
            mask = np.full(v,False)
            mask[edg-1] = True
        else:
            #if np.ndim(mask) > 1:
                #mask = np.squeeze(mask)
                #if mask.shape[0] > 1:
                #    mask = mask.T
            v = mask.size
        
        ## Compute the Lipschitz–Killing curvatures (LKC) - RV
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
            _, loc = ismember(tri[masktri,:][:,[0,1]], edg, 'rows')
            l12 = slm['resl'][loc,:]
            _, loc = ismember(tri[masktri,:][:,[0,2]], edg, 'rows')
            l13 = slm['resl'][loc,:]
            _, loc = ismember(tri[masktri,:][:,[1,2]], edg, 'rows')
            l23 = slm['resl'][loc,:]
            a = np.maximum(4*l12*l13-(l12+l13-l23)**2,0)
            r2 = np.mean(np.sqrt(a),axis=1)/4
            lkc[1,2] = np.sum(np.mean(np.sqrt(l12)+np.sqrt(l13)+np.sqrt(l23),axis=1))/2
            lkc[2,2] = np.sum(r2,axis=0)
        
            # Compute resels per mask vertex
            reselspvert = np.zeros(v)
            for j in range(0,3):
                reselspvert = reselspvert + np.bincount(tri[masktri,j],weights=r2,minlength=v)
            D = 2
            reselspvert = reselspvert.T / (D+1) / np.sqrt(4*np.log(2)) ** D
        else:
            reselspvert = None
        
    if 'lat' in slm:
        edg = SurfStatEdg(slm)
        # The lattice is filled with 5 alternating tetrahedra per cube
        I, J, K = np.shape(slm['lat'])
        IJ = I*J
        i, j = np.meshgrid(range(1,I+1),range(1,J+1))
        i = np.squeeze(np.reshape(i,(-1,1)))
        j = np.squeeze(np.reshape(j,(-1,1)))
        
        c1  = np.argwhere(((i+j)%2)==0 & (i < I) & (j < J))
        c2  = np.argwhere(((i+j)%2)==0 & (i > 1) & (j < J))
        c11 = np.argwhere(((i+j)%2)==0 & (i == I) & (j < J))
        c21 = np.argwhere(((i+j)%2)==0 & (i == I) & (j > 1))
        c12 = np.argwhere(((i+j)%2)==0 & (i < I) & (j == J))
        c22 = np.argwhere(((i+j)%2)==0 & (i > 1) & (j == J))

        d1  = np.argwhere(((i+j)%2)==0 & (i < I) & (j < J))+IJ
        d2  = np.argwhere(((i+j)%2)==0 & (i > 1) & (j < J))+IJ

        import numpy.concatenate as cat
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
        
        v = np.sum(slm.lat)
        if mask is None:
            mask = np.ones(1,v)
        
        reselspvert = np.zeros(v)
        vs = np.cumsum(np.squeeze(np.sum(np.sum(slm['lat'],axis=0),axis=1)))
        vs = cat((0,vs,vs(K)),axis=0)
        es = 0 
        lat = np.zeros((I,J,2))
        lat[:,:,0] = slm['lat'][:,:,0]
        lkc = np.zeros((4,4))
        n10 = np.floor(K/10)
        for k in range(0,K):
            f = k % 2
            if k < K:
                lat[:,:,f+1] = slm['lat'][:,:,k+1]
            else:
                lat[:,:,f+1] = np.zeros((I,J))
        vid = int(np.cumsum(lat) * np.reshape(lat.T,-1))

        
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



