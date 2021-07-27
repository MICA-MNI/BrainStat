"""Interpolations on a mesh."""
import logging

import nibabel as nib
import numpy as np
import trimesh
from brainspace.mesh.mesh_elements import get_cells, get_points
from brainspace.mesh.mesh_io import read_surface
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from scipy.interpolate.ndgriddata import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree


def surface_to_volume(
    pial_mesh,
    wm_mesh,
    labels,
    volume_template,
    volume_save,
    interpolation="nearest",
):
    """Projects surface labels to the cortical ribbon.

    Parameters
    ----------
    pial_mesh : str, BSPolyData
        Filename of a pial mesh or a BSPolyData object of the same.
    wm_mesh : str, BSPolyData
        Filename of a pial mesh or a BSPolyData object of the same.
    labels : str, numpy.ndarray
        Filename of a .label.gii or .shape.gii file, or a numpy array
        containing the labels.
    volume_template : str, nibabel.nifti1.Nifti1Image
        Filename of a nifti image in the same space as the mesh files or a
        NIfTI image loaded with nibabel.
    volume_save : str
        Filename to which the label image will be saved.
    interpolation : str
        Either 'nearest' for nearest neighbor interpolation, or 'linear'
        for trilinear interpolation, defaults to 'nearest'.
    """

    if not isinstance(pial_mesh, BSPolyData):
        pial_mesh = read_surface(pial_mesh)
    if not isinstance(wm_mesh, BSPolyData):
        wm_mesh = read_surface(wm_mesh)
    if not isinstance(volume_template, nib.nifti1.Nifti1Image):
        volume_template = nib.load(volume_template)

    logging.debug("Computing voxels inside the cortical ribbon.")
    ribbon_points = cortical_ribbon(pial_mesh, wm_mesh, volume_template)

    logging.debug("Computing labels for cortical ribbon voxels.")
    ribbon_labels = ribbon_interpolation(
        pial_mesh,
        wm_mesh,
        labels,
        volume_template,
        ribbon_points,
        interpolation=interpolation,
    )

    logging.debug("Constructing new nifti image.")
    new_data = np.zeros(volume_template.shape)
    ribbon_points = np.rint(
        ribbon_points, np.ones(ribbon_points.shape, dtype=int), casting="unsafe"
    )
    for i in range(ribbon_labels.shape[0]):
        new_data[
            ribbon_points[i, 0], ribbon_points[i, 1], ribbon_points[i, 2]
        ] = ribbon_labels[i]

    new_nii = nib.Nifti1Image(new_data, volume_template.affine)
    nib.save(new_nii, volume_save)


def cortical_ribbon(pial_mesh, wm_mesh, nii, mesh_distance=6):
    """Finds voxels inside of the cortical ribbon.

    Parameters
    ----------
    pial_mesh : BSPolyData
        Pial mesh.
    wm_mesh : BSPolyData
        White matter mesh.
    nii : Nibabel nifti
        Nifti image containing the space in which to output the ribbon.
    mesh_distance : int, optional
        Maximum distance from the cortical mesh at which the ribbon may occur.
        Used to reduce the search space, by default 6.

    Returns
    -------
    numpy.array
        Matrix coordinates of voxels inside the cortical ribbon.
    """

    try:
        import pyembree
    except ImportError:
        ModuleNotFoundError(
            "The package pyembree is required for this function. "
            + "You can install it with the conda package manager: "
            + "`conda install -c conda-forge pyembree`."
        )

    # Get world coordinates.
    x, y, z, _ = np.meshgrid(
        range(nii.shape[0]), range(nii.shape[1]), range(nii.shape[2]), 0
    )

    points = np.reshape(np.concatenate((x, y, z), axis=3), (-1, 3), order="F")
    world_coord = nib.affines.apply_affine(nii.affine, points)

    logging.debug("Discarding points that exceed the minima/maxima of the pial mesh.")
    # Discard points that exceed any of the maxima/minima
    pial_points = np.array(get_points(pial_mesh))
    discard = np.any(
        # If points exceed maximum coordinates
        (world_coord > np.amax(pial_points, axis=0)) |
        # If points are lower than minimum coordinates
        (world_coord < np.amin(pial_points, axis=0)),
        axis=1,
    )
    world_coord = world_coord[np.logical_not(discard), :]

    # Discard points that are more than mesh_distance from the pial and wm mesh.
    logging.debug("Discarding points that are too far from the meshes.")
    tree = cKDTree(pial_points)
    mindist_pial, _ = tree.query(world_coord)

    wm_points = np.array(get_points(wm_mesh))
    tree = cKDTree(wm_points)
    mindist_wm, _ = tree.query(world_coord)

    world_coord = world_coord[
        (mindist_pial < mesh_distance) & (mindist_wm < mesh_distance), :
    ]

    # Check which points are inside pial but not inside WM (i.e. ribbon)
    logging.debug("Retaining only points that are inside the pial but not the WM mesh.")
    pial_trimesh = trimesh.ray.ray_pyembree.RayMeshIntersector(
        trimesh.Trimesh(
            vertices=np.array(get_points(pial_mesh)),
            faces=np.array(get_cells(pial_mesh)),
        )
    )
    wm_trimesh = trimesh.ray.ray_pyembree.RayMeshIntersector(
        trimesh.Trimesh(
            vertices=np.array(get_points(wm_mesh)), faces=np.array(get_cells(wm_mesh))
        )
    )

    inside_wm = wm_trimesh.contains_points(world_coord)
    inside_pial = pial_trimesh.contains_points(world_coord)
    inside_ribbon = world_coord[inside_pial & ~inside_wm, :]
    ribbon_points = nib.affines.apply_affine(np.linalg.inv(nii.affine), inside_ribbon)
    return ribbon_points


def ribbon_interpolation(
    pial_mesh, wm_mesh, labels, nii, points, interpolation="nearest"
):
    """Performs label interpolation in the cortical ribbon.

    Parameters
    ----------
    pial_mesh : BSPolyData
        Pial mesh.
    wm_mesh : BSPolydata
        White matter mesh.
    labels : str, numpy.ndarray
        Filename of a .label.gii or .shape.gii file, or a numpy array
        containing the labels.
    nii : Nibabel nifti
        Reference nifti image.
    points : numpy.array
        Numpy array containing the coordinates of the ribbon.

    Returns
    -------
    numpy.array
        Interpolated value for each input point.

    Notes
    -----
    Strictly, this function will work outside the cortical ribbon too and assign
    any point to its label on the nearest mesh. An adventurous user could use
    this for nearest neighbour surface to volume anywhere in the brain, although
    such usage is not offically supported.
    """

    if not isinstance(labels, np.ndarray):
        labels = nib.gifti.giftiio.read(labels).agg_data()

    mesh_coord = np.concatenate((get_points(pial_mesh), get_points(wm_mesh)), axis=0)

    # Repeat labels as we concatenate the pial/white meshes.
    labels = np.concatenate((labels, labels))

    ribbon_coord = nib.affines.apply_affine(nii.affine, points)

    if interpolation == "nearest":
        interp = NearestNDInterpolator(mesh_coord, labels)
    elif interpolation == "linear":
        interp = LinearNDInterpolator(mesh_coord, labels)
    else:
        ValueError("Unknown interpolation type.")

    return interp(ribbon_coord)
