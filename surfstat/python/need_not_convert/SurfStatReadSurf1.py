"""
python version of SurfStatReadSurf1
"""

# Author: RRC
# License: BSD 3 clause


import numpy as np


@deprecated("BrainSpace dependency")
def py_SurfStatReadSurf1(filename, ab='a', numfields=4):
    """Reads coordinates and triangles from a single .obj or FreeSurfer file (mgz, gii, obj).

    Parameters
    ----------
    Usage: [ surf, ab ] = SurfStatReadSurf1( filename [,ab [,numfields ]] );

    filename : string
        .obj or FreeSurfer file name.
    ab : string
        'a' for ASCII or 'b' for binary. If it doesn't work it
        will try the other. Default is 'a'. Ignored if FS file.
    numfields : int
        number of fields to read, in the order below, default 4.

    Returns
    -------
    surf.coord  : 2D ndarray, shape = (n_coordinates, n_vertices)
        3 x v matrix of coordinates, v=#vertices.
    surf.normal : 2D ndarray, shape = (n_coordinates, n_vertices)
        3 x v matrix of surface normals, only .obj files.
    surf.colr : 2D ndarray, shape = (n_samples, n_feat)
        4 x 1 vector of colours for the whole surface,
        or 4 x v matrix of colours for each vertex, either
        uint8 in [0 255], or float in [0 1], only .obj files.
    surf.tri : tuple of BSPolyData or BSPolyData
        t x 3 matrix of triangle indices, 1-based, t=#triangles.
    ab : tuple of BSPolyData or BSPolyData
        whichever was successful.
    """

    sys.exit("Function py_SurfStatReadSurf1 is under development")
