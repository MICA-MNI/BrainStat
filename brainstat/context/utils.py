"""Utilities for handling label files"""

import os
import nibabel as nib
import numpy as np
import tempfile
import gzip
import shutil
from brainspace.mesh.mesh_io import read_surface

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
        if i is 0:
            img = nii.get_fdata()
            affine = nii.affine
            header = nii.header
        else:
            img[img==0] = nii.get_fdata()[img==0]
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

    if label_file.endswith('.gii'):
        labels = nib.gifti.giftiio.read(label_file).agg_data()
    elif label_file.endswith('.csv'):
        labels = np.loadtxt(label_file)
    else:
        ValueError('Unrecognized label file type.')

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
    if filename.endswith('.gz'):
        extension = os.path.splitext(filename[:-3])[-1]
        with tempfile.NamedTemporaryFile(suffix=extension) as f_tmp:
            with gzip.open(filename, 'rb') as f_gz:
                shutil.copyfileobj(f_gz, f_tmp)
            return read_surface(f_tmp.name)
    else:
        return read_surface(filename)