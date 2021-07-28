"""Utilities for handling label files"""

import gzip
import os
import shutil
import tempfile
from typing import List, Tuple, Union

import nibabel as nib
import numpy as np
from brainspace.mesh.mesh_io import read_surface
from brainspace.vtk_interface.wrappers.data_object import BSPolyData

from brainstat.mesh.interpolate import surface_to_volume

valid_surfaces = Union[
    str,
    BSPolyData,
    List[Union[str, BSPolyData]],
    Tuple[Union[str, BSPolyData], ...],
]


def multi_surface_to_volume(
    pial: valid_surfaces,
    white: valid_surfaces,
    volume_template: Union[str, nib.nifti1.Nifti1Image],
    output_file: str,
    labels: Union[str, np.ndarray, List[Union[np.ndarray, str]]],
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
    labels : str, numpy.ndarray, list, tuple
        Path to a label file for the surfaces, numpy array containing the
        labels, or a list containing multiple of the aforementioned.
    output_file: str
        Path to the output file, must end in .nii or .nii.gz.
    volume_template : str, nibabel.nifti1.Nifti1Image
        Path to a nifti file to use as a template for the surface to volume
        procedure, or a loaded NIfTI image.
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
        ValueError("Pial and white must be of the same type.")

    pial_list = _input_to_list(pial)
    white_list = _input_to_list(white)
    labels_list = _input_to_list(labels)

    if len(pial_list) is not len(white):
        ValueError("The same number of pial and white surfces must be provided.")

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
        ValueError("Unrecognized label file type.")

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


def _input_to_list(x: valid_surfaces) -> List[Union[str, BSPolyData]]:
    if isinstance(x, str):
        return [x]
    else:
        return list(x)
