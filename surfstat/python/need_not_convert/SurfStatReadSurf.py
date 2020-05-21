"""
python version of SurfStatReadSurf
"""

# Author: RRC
# License: BSD 3 clause


import numpy as np

@deprecated("BrainSpace dependency")
def py_SurfStatReadSurf(filenames, ab='a', numfields=2, dirname, maxmem=64):
    """Reads coordinates and triangles from an array of .obj or FreeSurfer files.

    Parameters
    ----------
    filenames : string
        .obj or FS file name (n=1) or n x k cell array of file names.
    ab : string
        'a' for ASCII or 'b' for binary. If it doesn't work it
            will try the other. Default is 'a'.
    numfields : int
        number of fields to read, in the order below, default 2.
    dirname : string
        name of a directory where you have write permission. This
        is only needed if the data is memory mapped. The data is
        memory mapped only if it exceeds maxmem Mb as 4 byte reals.
        SurfStatReadSurf then writes the data as 4 byte reals to a
        temporary file in dirname, then memory maps this file to the
        output data. If dirname does not exist, SurfStatReadVol will
        create it. The default is:
        (filenames{1,1} directory)/SurfStat, so you can ignore this
        parameter if you have write permission in the filenames{1,1}
        directory.
    maxmem : int, optional
        memory limit in Mb. The default is 64.

    Returns
    -------
    surf : tuple of BSPolyData or BSPolyData
    surf.tri : 2D ndarray, shape = (n_triangles, n_vertices)
        t x 3 matrix of triangle indices, 1-based, t=#triangles.
    surf.coord : 2D ndarray, shape = (n_coordinates, n_vertices)
        3 x v matrix of coordinates, v=#vertices, if n=1, or
        n x v x 3 array if n>1, or memory map of same. Data from the
        k files are concatenated. Note that the mapped file is not
        deleted after you quit MATLAB.
    ab : tuple of BSPolyData or BSPolyData
        whichever was successful.
        The coordinates and triangle indices of the k files are concatenated.
        If n=1, surf.coord is double precision;
        if n>1, surf.coord is single precision.
        surf.tri is int32.
    """

    sys.exit("Function py_SurfStatReadSurf is under development")
