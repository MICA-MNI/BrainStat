import numpy as np


def py_SurfStatInd2Coord(ind, surf):
    """Converts a vertex index to x,y,z coordinates

    Parameters
    ----------
    ind : 2D numpy array of shape (1,c)
        indices of vertex, 1-based
    surf : a dictionary with key 'coord' OR 'lat', 'vox', 'origin'
        surf['coord'] : 2D numpy array of shape (3,v), the coordinates.
        or
        surf['lat'] : 3D numpy array of 1's and 0's (1=in, 0=out).
        surf['vox'] : 2D numpy array of shape (1,3),
            voxel sizes in mm, [1,1,1] by default.
        surf['origin'] : 2D numpy array of shape (1,3),
            position of the first voxel in mm, [0,0,0] by default.

    Returns
    -------
    coord : 2D numpy array of shape (3,c)
        array of coordinates
    """

    ind = ind.astype(int)

    if 'coord' in surf.keys():
        coord = surf['coord'][:,ind-1].squeeze()

    if 'lat' in surf.keys():
        if not 'vox' in surf:
            surf['vox'] = np.ones((1,3))
        if not 'origin' in surf.keys():
            surf['origin'] = np.zeros((1,3))

        vid = np.cumsum(surf['lat'].T.flatten()) * surf['lat'].T.flatten()

        # implement matlab-ismember
        loc = []
        for i in range(0, ind.shape[1]):
            loc.append(np.where(vid == ind[0,i])[0].tolist())
        loc_flat = [item for sublist in loc for item in sublist]

        dim = np.shape(surf['lat'])
        i, j, k = np.unravel_index(loc_flat, dim, order='F')

        coord = np.zeros((3,  ind.shape[1]))
        coord[0,:] = surf['origin'][0,0] + np.multiply(i, surf['vox'][0,0])
        coord[1,:] = surf['origin'][0,1] + np.multiply(j, surf['vox'][0,1])
        coord[2,:] = surf['origin'][0,2] + np.multiply(k, surf['vox'][0,2])

    return coord
