"""Interpolations on a mesh."""
import gzip
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Union

import nibabel as nib
import numpy as np
import trimesh
from brainspace.mesh.mesh_elements import get_cells, get_points
from brainspace.mesh.mesh_io import read_surface
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from nilearn.datasets import load_mni152_brain_mask
from scipy.interpolate.ndgriddata import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

from brainstat._utils import data_directories
from brainstat.datasets import fetch_template_surface


def surface_to_volume(
    pial_mesh: Union[str, BSPolyData],
    wm_mesh: Union[str, BSPolyData],
    labels: Union[str, np.ndarray],
    volume_template: Union[str, nib.nifti1.Nifti1Image],
    volume_save: str,
    interpolation: str = "nearest",
) -> None:
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


valid_surfaces = Union[
    str,
    BSPolyData,
    Sequence[Union[str, BSPolyData]],
]


def _input_to_list(x: valid_surfaces) -> List[Union[str, BSPolyData]]:
    if isinstance(x, str):
        return [x]
    else:
        return list(x)


def multi_surface_to_volume(
    pial: valid_surfaces,
    white: valid_surfaces,
    volume_template: Union[str, nib.nifti1.Nifti1Image],
    output_file: str,
    labels: Union[str, np.ndarray, Sequence[Union[np.ndarray, str]]],
    interpolation: str = "nearest",
) -> None:
    """Interpolates multiple surfaces to the volume.

    Parameters
    ----------
    pial : str, BSPolyData, list, tuple
        Path of a pial surface file, BSPolyData of a pial surface or a list
        containing multiple of the aforementioned.
    white : str, BSPolyData, list, tuple
        Path of a white matter surface file, BSPolyData of a pial surface or a
        list containing multiple of the aforementioned.
    volume_template : str, nibabel.nifti1.Nifti1Image
        Path to a nifti file to use as a template for the surface to volume
        procedure, or a loaded NIfTI image.
    output_file: str
        Path to the output file, must end in .nii or .nii.gz.
    labels : str, numpy.ndarray, list, tuple
        Path to a label file for the surfaces, numpy array containing the
        labels, or a list containing multiple of the aforementioned.
    interpolation : str
        Either 'nearest' for nearest neighbor interpolation, or 'linear'
        for trilinear interpolation, defaults to 'nearest'.

    Notes
    -----
    An equal number of pial/white surfaces and labels must be provided. If
    parcellations overlap across surfaces, then the labels are kept for the
    first provided surface.
    """

    # Deal with variety of ways to provide input.
    if type(pial) is not type(white):
        raise ValueError("Pial and white must be of the same type.")

    pial_list = _input_to_list(pial)
    white_list = _input_to_list(white)
    labels_list = _input_to_list(labels)

    if len(pial_list) is not len(white):
        raise ValueError("The same number of pial and white surfces must be provided.")

    for i in range(len(pial_list)):
        if not isinstance(pial_list[i], BSPolyData):
            pial_list[i] = read_surface_gz(pial_list[i])

        if not isinstance(white_list[i], BSPolyData):
            white_list[i] = read_surface_gz(white_list[i])

    if not isinstance(volume_template, nib.nifti1.Nifti1Image):
        volume_template = nib.load(volume_template)

    for i in range(len(labels_list)):
        if isinstance(labels_list[i], np.bool_):
            labels_list[i] = np.array(labels_list[i])
        elif not isinstance(labels[i], np.ndarray):
            labels_list[i] = load_mesh_labels(labels_list[i])

    # Surface data to volume.
    T = []
    for i in range(len(pial)):
        T.append(tempfile.NamedTemporaryFile(suffix=".nii.gz"))
        surface_to_volume(
            pial_list[i],
            white_list[i],
            labels[i],
            volume_template,
            T[i].name,
            interpolation=interpolation,
        )

    if len(T) > 1:
        T_names = [x.name for x in T]
        combine_parcellations(T_names, output_file)
    else:
        shutil.copy(T[0].name, output_file)


def cortical_ribbon(
    pial_mesh: BSPolyData,
    wm_mesh: BSPolyData,
    nii: nib.nifti1.Nifti1Image,
    mesh_distance: float = 6,
) -> np.ndarray:
    """Finds voxels inside of the cortical ribbon.

    Parameters
    ----------
    pial_mesh : BSPolyData
        Pial mesh.
    wm_mesh : BSPolyData
        White matter mesh.
    nii : Nibabel nifti
        Nifti image containing the space in which to output the ribbon.
    mesh_distance : float, optional
        Maximum distance from the cortical mesh at which the ribbon may occur.
        Used to reduce the search space, by default 6.

    Returns
    -------
    numpy.ndarray
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
    pial_mesh: BSPolyData,
    wm_mesh: BSPolyData,
    labels: Union[str, np.ndarray],
    nii: nib.nifti1.Nifti1Image,
    points: np.ndarray,
    interpolation: str = "nearest",
) -> np.ndarray:
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
    interpolation : str, optional
        Interpolation method. Can be either 'nearest' or 'linear'.

    Returns
    -------
    numpy.ndarray
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


def __create_precomputed(data_dir: Optional[Union[str, Path]] = None) -> None:
    """Create nearest neighbor interpolation niftis for MATLAB."""
    data_dir = Path(data_dir) if data_dir else data_directories["BRAINSTAT_DATA_DIR"]
    mni152 = load_mni152_brain_mask()
    for template in ("fsaverage5", "fsaverage"):
        output_file = data_dir / f"nn_interp_{template}.nii.gz"
        if output_file.exists():
            continue
        pial = fetch_template_surface(template, layer="pial", join=False)
        white = fetch_template_surface(template, layer="white", join=False)
        labels = (
            np.arange(1, get_points(pial[0]).shape[0] + 1),
            np.arange(
                get_points(pial[0]).shape[0] + 1, get_points(pial[0]).shape[0] * 2 + 1
            ),
        )
        multi_surface_to_volume(
            pial=pial,
            white=white,
            volume_template=mni152,
            labels=labels,
            output_file=str(output_file),
            interpolation="nearest",
        )

    if not (data_dir / "nn_interp_hcp.nii.gz").exists():
        import hcp_utils as hcp
        from brainspace.mesh.mesh_creation import build_polydata

        pial_fslr32k = (build_polydata(hcp.mesh.pial[0], hcp.mesh.pial[1]),)
        white_fslr32k = (build_polydata(hcp.mesh.white[0], hcp.mesh.white[1]),)
        labels_fslr32k = (np.arange(1, get_points(pial[0]).shape[0] + 1),)
        multi_surface_to_volume(
            pial=pial,
            white=white,
            volume_template=mni152,
            labels=labels_fslr32k,
            output_file=str(data_dir / "nn_interp_fslr32k.nii.gz"),
            interpolation="nearest",
        )


def combine_parcellations(files: List[str], output_file: str) -> None:
    """Combines multiple nifti files into one.

    Parameters
    ----------
    files : list
        List of strings containing the paths to nifti files.
    output_file : str
        Path to the output file.

    Notes
    -----
    This function assumes that 0's are missing data. When multiple files have
    non-zero values in the same voxel, then the data from the first provided
    file is kept.
    """
    for i in range(len(files)):
        nii = nib.load(files[i])
        if i == 0:
            img = nii.get_fdata()
            affine = nii.affine
            header = nii.header
        else:
            img[img == 0] = nii.get_fdata()[img == 0]
    new_nii = nib.Nifti1Image(img, affine, header)
    nib.save(new_nii, output_file)


def load_mesh_labels(label_file: str, as_int: bool = True) -> np.ndarray:
    """Loads a .label.gii or .csv file.

    Parameters
    ----------
    label_file : str
        Path to the label file.
    as_int : bool
        Determines whether to enforce integer format on the labels, defaults to True.

    Returns
    -------
    numpy.ndarray
        Labels in the file.
    """

    if label_file.endswith(".gii"):
        labels = nib.gifti.giftiio.read(label_file).agg_data()
    elif label_file.endswith(".csv"):
        labels = np.loadtxt(label_file)
    else:
        raise ValueError("Unrecognized label file type.")

    if as_int:
        labels = np.round(labels).astype(int)
    return labels


def read_surface_gz(filename: str) -> BSPolyData:
    """Extension of brainspace's read_surface to include .gz files.

    Parameters
    ----------
    filename : str
        Filename of file to open.

    Returns
    -------
    BSPolyData
        Surface mesh.
    """
    if filename.endswith(".gz"):
        extension = os.path.splitext(filename[:-3])[-1]
        with tempfile.NamedTemporaryFile(suffix=extension) as f_tmp:
            with gzip.open(filename, "rb") as f_gz:
                shutil.copyfileobj(f_gz, f_tmp)
            return read_surface(f_tmp.name)
    else:
        return read_surface(filename)
