import numpy as np
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_creation import build_polydata
from brainspace.mesh.mesh_elements import get_cells, get_points
from brainspace.vtk_interface.wrappers.data_object import BSPolyData

def py_SurfStatAvSurf(filenames, fun=lambda x, y: x+y):
    """Average, minimum, or maximum of surfaces.

    Args:
        filenames (list, tuple): List/tuple of filenames of surfaces or BSPolyData objects.
        fun (lambda function): Lambda function to use on the surface coordinates. 
            Defaults to lambda x, y: x+y.

    Returns:
        surface [BSPolyData]: The output surface.
    """
    
    for i in range(0, len(filenames)):
        # Check whether input is BSPolyData or a filename. 
        if isinstance(filenames[i], BSPolyData):
            s = filenames[i]    
        else:
            s = read_surface(filenames[i])
        
        # Grab the triangles only from the first surface, 
        # apply function to coordinates.
        if i is 0:
            tri = get_cells(s) 
            coord = get_points(s)
            m = 1
        else:
            coord = fun(get_points(s), coord)
            m = fun(m,1)
    
    coord = coord / m 
    surface = build_polydata(coord, tri)
    return surface
            
            