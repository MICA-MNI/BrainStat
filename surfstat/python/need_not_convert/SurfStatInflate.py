"""
python version of SurfStatInflate
"""

# Author: RRC
# License: BSD 3 clause


import numpy as np

def py_SurfStatInflate(surf, w=0.5, spherefile):
    """Inflates a surface mesh to hemi-ellipsoids.

    Parameters
    ----------
    surf : tuple of BSPolyData or BSPolyData
        3 x v matrix of coordinates, v=#vertices.
    w : float
        weight in [0,1] given to hemi-ellipsoids, default 0.5.
    spherefile : tuple of BSPolyData or BSPolyData
        file name of a sphere surface for the left hemisphere. If
        it is a .obj file, it assumes that the triangulation of the
        right hemisphere is a mirror image; if it is an FS file,
        then it assumes it is identical. The default is sphere.obj
        for a 40962 vertex (v=81924 for both hemispheres)
        icosahedral mesh, or lh.sphere for a 163842 vertex
        (v=327684 for both hemispheres) icosahedral mesh.

    Returns
    -------
    surfw : tuple of BSPolyData or BSPolyData
        3 x v matrix of inflated coordinates.
    """


    sys.exit("Function py_SurfStatInflate is under development")
