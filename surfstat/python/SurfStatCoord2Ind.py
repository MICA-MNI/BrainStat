import numpy as np

def py_SurfStatCoord2Ind(coord, surf):
    """Converts a vertex index to x,y,z coordinates

    Parameters
    ----------
    coord : 2D numpy array of shape (c,3)
        coordinates for finfing indices thereof.
    surf : a dictionary with key 'coord' OR 'lat', 'vox', 'origin'
        surf['coord'] : 2D numpy array, the coordinates.
        or
        surf['lat'] : 3D numpy array of 1's and 0's (1=in, 0=out).
        surf['vox'] : 2D numpy array of shape (1,3),
            voxel sizes in mm, [1,1,1] by default.
        surf['origin'] : 2D numpy array of shape (1,3),
            position of the first voxel in mm, [0,0,0] by default.

    Returns
    -------
    ind : 2D numpy array of shape (c,1)
        indices of the nearest vertex to the surface. If surf us a volume and
        the point is outside, then ind = 0.
    """

    c = np.shape(coord)[0]
    ind = np.zeros((c,1))

    if 'coord' in surf.keys():
        v = np.shape(surf['coord'])[1]
        for i in range(0, c):
            dist =  np.square(surf['coord'] - np.tile(coord[i,:], (v,1)).T)\
                    .sum(axis=0) 
            ind[i] = np.array(np.where(dist == dist.min())) 
            ind[i] = ind[i] +1   

    if 'lat' in surf.keys():
        if not 'vox' in surf:
            surf['vox'] = np.ones((1,3))
        if not 'origin' in surf.keys():
            surf['origin'] = np.zeros((1,3))
            
        i = np.around((coord[:,0] - surf['origin'][0,0]) / (surf['vox'][0,0] + \
            int(surf['vox'][0,0]==0))+1)
        j = np.around((coord[:,1] - surf['origin'][0,1]) / (surf['vox'][0,1] + \
            int(surf['vox'][0,1]==0))+1)
        k = np.around((coord[:,2] - surf['origin'][0,2]) / (surf['vox'][0,2] + \
            int(surf['vox'][0,2]==0))+1)            
        
        dim = np.shape(surf['lat'])    
        i[np.logical_or((i<1), (i>dim[0]))] = 0
        j[np.logical_or((j<1), (j>dim[1]))] = 0
        k[np.logical_or((k<1), (k>dim[2]))] = 0

        a = [m and n and l for m, n, l, in zip(i,j,k)]
        a = np.array(a, dtype=bool)

        ind = np.zeros((c,1))
        vid = np.cumsum(surf['lat'].T.flatten()) * surf['lat'].T.flatten()
        values = []

        # check if indices are all not empty
        list_to_check = np.concatenate((i[a], j[a], k[a])).tolist()
        if list_to_check:       
            for row in range(0, len(a)):
                XI = [i[row]-1, j[row]-1, k[row]-1]
                XI = [int(xi) for xi in XI]
                values.append(vid[np.ravel_multi_index(XI, dim, order='F')])
            values = np.array(values).reshape(ind[a].shape)
            ind[a] = values         

    ind = ind.reshape(c, 1)
    return ind

