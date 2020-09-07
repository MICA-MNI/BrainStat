import sys
sys.path.append('matlab')
import numpy as np
import scipy.io as sio
import matlab.engine
import matlab
global eng
eng = matlab.engine.start_matlab()
addpath = eng.addpath('matlab')

def py_SurfStatInflate(surf, w=0.5, spherefile=None):
    """Inflates a surface mesh to hemi-ellipsoids.

    Parameters
    ----------
    surf : a dictionary with key 'coord'
        surf['coord'] : 2D numpy array of shape (3,v)
            Matrix of coordinates, v is the number of vertices.
    w : a float, by default 0.5,
        Weight in [0,1] given to hemi-ellipsoids.
    spherefile : string (a filename)
        Filename of a sphere surface for the left or right hemisphere. If 
        spherefile is None, and v=40962 (or v=81924), by default 'sphere.obj'. 
        If spherefile is None, and v=163842 (or v=327684), by default 
        'lh.sphere'. If spherefile is an *.obj file, it assumes that the 
        triangulation of the right hemisphere is a mirror image; if it is an 
        FS file, then it assumes it is identical. 

    Returns
    -------
    surfw : a dictionary with key 'coord',
        surfw['coord'] : 2D numpy array of shape (3,v),
            matrix of inflated coordinates.
    """
    
    v = surf['coord'].shape[1]
    
    if v <= 81924:
        # MATLAB RAPPING FOR *obj FILE READ IN --> has to be changed...
        if spherefile is None:
            spherefile = 'sphere.obj'
        sphere_mat = eng.SurfStatReadSurf(spherefile)
        sphere = {}
        sphere['tri'] = np.array(sphere_mat['tri']) 
        sphere['coord'] = np.array(sphere_mat['coord'])
    
        if v == 81924:
            sphere['tri'] = np.concatenate((sphere['tri'],
                                            sphere['tri']+v), axis=1)
            col1 = sphere['coord'][0,:] * (sphere['coord'][0,:] < 0)
            col2 = -1 *sphere['coord'][0,:] * (sphere['coord'][0,:] < 0)
            x = np.concatenate((col1,col2))
            x = x.reshape(1, len(x))
            row2 = row3 = sphere['coord'][1:3,:]
            y = np.concatenate((row2,row3), axis=1)
            sphere['coord'] = np.concatenate((x,y))
        else:
            if surf['coord'][0,:].mean()/abs(surf['coord'][0,:]).mean() <-0.5:
                row1 = sphere['coord'][0,:] * (sphere['coord'][0,:] < 0)
                row1 = row1.reshape(1, len(row1))
                sphere['coord'] = np.concatenate((row1,
                                                  sphere['coord'][1:3,:]))
            else:
                row1 = -sphere['coord'][0,:] * (sphere['coord'][0,:] < 0)                                                   
                row1 = row1.reshape(1, len(row1))
                sphere['coord'] = np.concatenate((row1,
                                                  sphere['coord'][1:3,:]))
    else:
        if spherefile is None:
            spherefile = 'lh.sphere'
        # MATLAB RAPPING FOR *sphere FILE READ IN --> has to be changed...
        sphere_mat = eng.SurfStatReadSurf(spherefile)
        sphere = {}
        sphere['tri'] = np.array(sphere_mat['tri'])
        sphere['coord'] = np.array(sphere_mat['coord'])
        
        if v == 327684:
            sphere['tri'] = np.concatenate((sphere['tri'],
                                            sphere['tri']+v), axis=1)
            col1 = sphere['coord'][0,:] * (sphere['coord'][0,:] < 0)
            col2 = sphere['coord'][0,:] * (sphere['coord'][0,:] > 0)
            x = np.concatenate((col1,col2))
            x = x.reshape(1, len(x))
            row2 = row3 = sphere['coord'][1:3,:]
            y = np.concatenate((row2,row3), axis=1)
            sphere['coord'] = np.concatenate((x,y))
        else:
            if surf['coord'][0,:].mean()/abs(surf['coord'][0,:]).mean() <-0.5:
                row1 = sphere['coord'][0,:] * (sphere['coord'][0,:] < 0)
                row1 = row1.reshape(1, len(row1))
                sphere['coord'] = np.concatenate((row1,
                                                  sphere['coord'][1:3,:]))
            else:
                row1 = sphere['coord'][0,:] * (sphere['coord'][0,:] > 0)
                row1 = row1.reshape(1, len(row1))
                sphere['coord'] = np.concatenate((row1,
                                                  sphere['coord'][1:3,:]))
    maxs = surf['coord'].max(1)
    mins = surf['coord'].min(1)
    maxsp = sphere['coord'].max(1)
    minsp = sphere['coord'].min(1)
    surfw = surf

    for i in range(0,3): 
        surfw['coord'][i,:] = ((sphere['coord'][i,:] - minsp[i]) / \
        (maxsp[i]-minsp[i]) * (maxs[i]-mins[i]) + mins[i]) * w + \
        surf['coord'][i,:]*(1-w) 

    return surfw
