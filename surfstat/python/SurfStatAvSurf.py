import numpy as np
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_creation import build_polydata
from brainspace.mesh.mesh_elements import get_cells, get_points
from brainspace.vtk_interface.wrappers.data_object import BSPolyData

def py_SurfStatAvSurf(filenames, fun=lambda x, y: x+y):
    """Average, minimum, or maximum of surfaces.

    Args:
        filenames (2D numpy array): Numpy array of filenames of surfaces or BSPolyData objects.
        fun (lambda function): Lambda function to use on the surface coordinates. 
            Defaults to lambda x, y: x+y.

    Returns:
        surface [BSPolyData]: The output surface.
    """
    if filenames.ndim is not 2:
        raise ValueError('Filenames must be a 2-dimensional array.')
      
    for i in range(0, filenames.shape[0]):
        surfaces = np.empty(filenames.shape[1], dtype=np.object)
        for j in range(0, filenames.shape[1]):
            
            # Check whether input is BSPolyData or a filename. 
            if isinstance(filenames[i,j], BSPolyData):
                surfaces[j] = filenames[i,j]    
            else:
                surfaces[j] = read_surface(filenames[i,j])
            
            # Concatenate second dimension of filenames. 
            if j is 0:
                tri = get_cells(surfaces[j]) 
                coord = get_points(surfaces[j])
            else:
                tri = np.concatenate((tri, get_cells(surfaces[j]) + coord.shape[0]), axis=0)
                coord = np.concatenate((coord, get_points(surfaces[j])), axis=0)
            
        if i is 0:
            m = 1
            coord_all = coord
        else:
            coord_all = fun(coord_all,coord)
            m = fun(m,1)
    
    coord_all = coord_all / m 
    surface = build_polydata(coord_all, tri)
    return surface
            
            