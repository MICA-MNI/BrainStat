"""
python version of SurfStatROI.
"""

# Author: RRC
# License: BSD 3 clause


import numpy as np

def py_SurfStatROI(centre, radius, surf):
    """ROI on the surface or volume with a given centre and radius

    Parameters
    ----------
    centre : 2D ndarray, shape = (n_coordinates, length)
        id number of vertex, or
        3 x 1 vector of [x; y; z] coordinates in mm
    radius : float
        radius, mm.
    surf.coord : 2D ndarray, shape = (n_coordinates, n_vertices)
        3 x v matrix of coordinates of surface.
        or
    surf.lat :
        nx x ny x nz array, 1=in, 0=out, clamped to the mask
    surf.vox :
        1 x 3 vector of voxel sizes in mm of the clamped mask
    surf.origin :
        position in mm of the first voxel of the clamped mask

    Returns
    -------
    maskROI :  1D ndarray, shape = (1, n_vertices)
        1 x v vector, 1=inside ROI, 0=outside
    """
    sys.exit("Function py_SurfStatROI is under development")
