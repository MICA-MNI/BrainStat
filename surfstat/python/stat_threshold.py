import math
import numpy as np
from scipy.special import betaln, gammaln, gamma

def gammalni(n):
    x = math.inf * np.ones(n.shape)
    x[n>=0] = gammaln(n[n>=0])
    return x

def stat_threshold(search_volume=0, num_voxels=1, fwhm=0.0, df=math.inf, 
    p_val_peak=0.05, cluster_threshold=0.001, p_val_extent=0.05, nconj=1, 
    nvar=1, EC_file=None, expr=None, nprint=5):

    ## Deal with the input

    # Make sure all input is in np.array format.
    fwhm = np.array(fwhm, ndmin=1)
    search_volume = np.array(search_volume, ndmin=1)
    num_voxels = np.array(num_voxels)
    df = np.array(df)
    nvar = np.array(nvar)

    # Set the FWHM
    if fwhm.ndim == 1:
        fwhm = np.expand_dims(fwhm,axis=0)
        fwhm = np.r_[fwhm,fwhm]
    if np.shape(fwhm)[1] == 1:
        scale = 1
    else:
        scale = fwhm[0,1] / fwhm[0,0]
        fwhm = fwhm[:,0]
    isscale = scale > 1

    # Set the number of voxels
    if num_voxels.size == 1:
        num_voxels = np.append(num_voxels,1)
    
    # Set the search volume.
    if search_volume.ndim == 1:
        search_volume = np.expand_dims(search_volume,1)
        radius = (search_volume / (4/3*math.pi)) ** (1/3)
        search_volume = np.c_[np.ones(radius.shape),
                              4 * radius,
                              2 * radius ** 2 * math.pi,
                             search_volume]

    if search_volume.shape[0] == 1:
        search_volume = np.concatenate((search_volume, 
                                        np.concatenate((np.ones((1,1)),np.zeros((1,search_volume.size-1))),axis=1)),
                                        axis=0)
    
    lsv = search_volume.shape[1]
    if all(fwhm>0):
        fwhm_inv = all(fwhm>0) / fwhm + any(fwhm<=0)
    else:
        fwhm_inv = np.zeros(fwhm.shape)
    resels = search_volume * fwhm_inv ** np.arange(0,lsv)
    invol = resels * (4*math.log(2)) ** (np.arange(0,lsv)/2)

    D = invol.shape[1] - np.argmax(np.fliplr(invol),axis=1) - 1

    # determines which method was used to estimate fwhm (see fmrilm or multistat): 
    df_limit=4

    if df.size == 1:
        df = np.c_[df,np.zeros((1,1))]
    if df.shape[0] == 1:
        infs = np.array([math.inf,math.inf],ndmin=2)
        df = np.r_[df, infs, infs]
    if df.shape[1] == 1:
        df = np.c_[df,df]
        df[0,1] = 0
    if df.shape[0] == 2:
        df = np.r_[df,np.expand_dims(df[1,:],axis=0)]

    # is_tstat=1 if it is a t statistic
    is_tstat = df[0,1]==0
    if is_tstat:
        df1 = 1
        df2 = df[0,0]
    else:
        df1 = df[0,0]
        df2 = df[0,1]
    if df2 >= 1000:
        df2 = math.inf
    df0 = df1 + df2

    dfw1 = df[1:3,1]
    dfw2 = df[1:3,2]

    dfw1[dfw1 >= 1000] = math.inf
    dfw2[dfw2 >= 1000] = math.inf

    if nvar.size == 1:
        nvar = np.c_[nvar,df1]
    
    if isscale and (D[1]>1 or nvar[0,0] > 1 | df2 < math.inf):
        print(D)
        print(nvar)
        print(df2)
        print('Cannot do scale space.')
        return
    Dlim = D + np.array([scale > 1, 0])
    DD = Dlim + nvar - 1

    # Values of the F statistic: 
    t = (np.arange(1000,0,-1)/100) ** 4

    # Find the upper tail probs cumulating the F density using Simpson's rule:
    if math.isinf(df2):
        u = df1*t
        b = math.exp(-u/2-math.log(2*math.pi)/2+math.log(u)/4)*df1**(1/4)*4/100
    else:
        u = df1*t/df2
        b=math.exp(-df0/2*math.log(1+u)+math.log(u)/4-betaln(1/2, (df0-1)/2))*(df1/df2)**(1/4)*4/100

    t = np.r_[t,0]
    b = np.r_[b,0]
    n = t.size 
    sb = np.cumsum(b)
    sb1 = np.cumsum(b * (-1) ** np.arange(1,n+1))
    pt1 = sb + sb1/3 - b/3
    pt2 = sb - sb1/3 - b/3
    tau = np.zeros(n, DD[0]+1, DD[1]+1)
    tau[0:n:2,1,1] = pt1[0:n:2]
    tau[1:n:2,1,1] = pt2[1:n:2]
    tau[n-1,0,0] = 1
    tau[:,1,1] = np.min(tau[:,1,1])

    # Find the EC densities:
    u = df1 * t
    for d in range(1,np.max(DD)+1):
        for e in range(0,np.min(DD)+1):
            s1 = 0
            cons = -((d+e)/2+1)*math.log(math.pi)+gammaln(d)+gammaln(e+1)
            for k in np.arange(0,(d-1+e)/2+1):
                i, j = np.meshgrid(np.arange(0,k+1),np.arange(0,k+1))
                if df2 == math.inf:
                    q1 = math.log(math.pi)/2-((d+e-1)/2+i+j)*math.log(2)
                else:
                    q1 = (df0-1-d-e)*math.log(2)+gammaln((df0-d)/2+i)+gammaln((df0-e)/2+j)-gammalni(df0-d-e+i+j+k)-((d+e-1)/2-k)*math.log(df2)
                q2=cons-gammalni(i+1)-gammalni(j+1)-gammalni(k-i-j+1)-gammalni(d-k-i+j)-gammalni(e-k-j+i+1)
                s2 = np.sum(math.exp(q1+q2))
                if s2 > 0:
                    s1=s1+(-1)**k*u**((d+e-1)/2-k)*s2
            
            if df2 == math.inf:
                s1 = s1 * math.exp(-u/2)
            else:
                s1 = s1 * math.exp(-(df0-2)/2*math.log(1+u/df2))
            
            if DD(1) >= DD(2):
                tau[:,d,e] = s1
                if d <= np.min(DD):
                    tau[:,e,d] = s1
            else:
                tau[:,e,d] = s1
                if d<= np.min(DD):
                    tau[:,d,e] = s1
    
    # For multivariate statistics, add a sphere to the search region:
    a = np.zeros((2,np.max(nvar)))
    for k in range(0,2):
        j = np.arange((nvar[k]-1),-0.001,-2)
        a[k,j] = math.exp(j*math.log(2)+j/2*math.log(math.pi) + 
            gammaln((nvar[k]+1)/2)-gammaln((nvar[k]+1-j)/2)-gammaln(j+1))

    rho = np.zeros((n, Dlim[0]+1, Dlim[1]+1))

    for k in range(0,nvar[0]):
        for l in range(0,nvar[1]):
            rho = rho + a[0,k] * a[1,l] * tau[:, np.arange(0,Dlim[0])+k, np.arange(0,Dlim[1])+l]
    
    if is_tstat:
        if all(nvar==1):
            t = np.r_[np.sqrt(t[0:n]), -np.sqrt(t)[::-1]]
            rho = np.r_[rho[0:n,:,:], rho[::-1]/2]
            for i in range(0,D[0]+1):
                for j in range(0,D[1]+1):
                    rho[n-1+np.arange(0,n),i,j] = -(-1)**(i+j)*rho[n-1+np.arange(0,n),i,j]
            rho[n-1+np.arange(0,n),0,0] = rho[n-1+np.arange(0,n),0,0] + 1
            n = 2 * n-1
        else:
            t = np.sqrt(t)
    
    # For scale space.
    if scale > 1:
        kappa = D[0]/2
        tau = np.zeros(n,D[0]+1)
        for d in range(0,D[0]+1):
            s1 = 0
            for k in range(0,d/2+1):
                s1 = s1+(-1)^k/(1-2*k)*math.exp(gammaln(d+1)-gammaln(k+1)-gammaln(d-2*k+1)
                    + (1/2-k)*math.log(kappa)-k*math.log(4*math.pi)) * rho[:,d+1-2*k,1]
            if d == 0:
                cons = math.log(scale)
            else:
                cons = (1-1/scale**d)/d
            tau[:,d] = rho[:,d,1] * (1+1/scale**d) / 2 + s1 * cons
        rho[:,0:D[1],0] = tau
    
    if D[1] == 0:
        d = D[0]
        if nconj > 1:
            # Conjunctions
            b = gamma((np.arange(1,d+2)/2) / gamma(1/2))






