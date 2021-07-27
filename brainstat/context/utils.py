"""Utilities for handling label files"""

import gzip
import os
import shutil
import tempfile

import nibabel as nib
import numpy as np
from brainspace.mesh.mesh_io import read_surface
from brainspace.vtk_interface.wrappers.data_object import BSPolyData

from brainstat.mesh.interpolate import surface_to_volume


def multi_surface_to_volume(
    pial,
    white,
    volume_template,
    labels,
    output_file,
    interpolation="nearest",
):
    """Interpolates multiple surfaces to the volume.

    Parameters
    ----------
    pial : str, BSPolyData, list
        Path of a pial surface file, BSPolyData of a pial surface or a list
        containing multiple of the aforementioned.
    white : str, BSPolyData, list
        Path of a white matter surface file, BSPolyData of a pial surface or a
        list containing multiple of the aforementioned.
    labels : str, numpy.ndarray, list
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

    if isinstance(pial, tuple):
        pial = list(pial)
        white = list(white)
    elif not isinstance(pial, list):
        pial = [pial]
        white = [white]

    if isinstance(labels, tuple):
        labels = list(labels)
    elif not isinstance(labels, list):
        labels = [labels]

    if len(pial) is not len(white):
        ValueError("The same number of pial and white surfces must be provided.")

    for i in range(len(pial)):
        if not isinstance(pial[i], BSPolyData):
            pial[i] = read_surface_gz(pial[i])

        if not isinstance(white[i], BSPolyData):
            white[i] = read_surface_gz(white[i])

    if not isinstance(volume_template, nib.nifti1.Nifti1Image):
        volume_template = nib.load(volume_template)

    for i in range(len(labels)):
        if isinstance(labels[i], np.bool_):
            labels[i] = np.array(labels[i])
        elif not isinstance(labels[i], np.ndarray):
            labels[i] = load_mesh_labels(labels[i])

    # Surface data to volume.
    T = []
    for i in range(len(pial)):
        T.append(tempfile.NamedTemporaryFile(suffix=".nii.gz"))
        surface_to_volume(
            pial[i],
            white[i],
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


def combine_parcellations(files, output_file):
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


def load_mesh_labels(label_file, as_int=True):
    """Loads a .label.gii or .csv file.

    Parameters
    ----------
    label_file : str
        Path to the label file.
    as_int : bool
        Determines whether to enforce integer format on the labels, defaults to True.

    Returns
    -------
    numpy.array
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


def read_surface_gz(filename):
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
