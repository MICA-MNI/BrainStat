""" Meta-analytic decoding based on NiMARE """
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import nimare
import numpy as np
import pandas as pd
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from neurosynth.base.dataset import Dataset, download
from nibabel.nifti1 import NiftiImage
from nilearn.datasets import load_mni152_brain_mask
from nilearn.input_data import NiftiMasker

from .utils import multi_surface_to_volume


def surface_decode_nimare(
    pial: Union[str, BSPolyData, Sequence[Union[str, BSPolyData]]],
    white: Union[str, BSPolyData, Sequence[Union[str, BSPolyData]]],
    stat_labels: Union[str, np.ndarray, Sequence[Union[str, np.ndarray]]],
    mask_labels: Union[str, np.ndarray, Sequence[Union[str, np.ndarray]]],
    interpolation: str = "linear",
    data_dir: str = None,
    feature_group: str = None,
    features: Sequence[str] = None,
) -> pd.DataFrame:
    """Meta-analytic decoding of surface maps using NeuroSynth or Brainmap.

    Parameters
    ----------
    pial : str, BSPolyData, sequence of str or BSPolyData
        Path of a pial surface file, BSPolyData of a pial surface or a list
        containing multiple of the aforementioned.
    white : str, BSPolyData, sequence of str or BSPolyData
        Path of a white matter surface file, BSPolyData of a pial surface or a
        list containing multiple of the aforementioned.
    stat_labels : str, numpy.ndarray, sequence of str or numpy.ndarray
        Path to a label file for the surfaces, numpy array containing the
        labels, or a list containing multiple of the aforementioned.
    mask_labels : str, numpy.ndarray, sequence of str of or numpy.ndarray
        Path to a mask file for the surfaces, numpy array containing the
        mask, or a list containing multiple of the aforementioned. If None
        all vertices are included in the mask. Defaults to None.
    interpolation : str, optional
        Either 'nearest' for nearest neighbor interpolation, or 'linear'
        for trilinear interpolation, by default 'linear'.
    data_dir : str, optional
        The directory of the nimare dataset. If none exists, a new dataset will
        be downloaded and saved to this path. If None, the directory defaults to
        your home directory, by default None.
    correction : str, optional
        Multiple comparison correction. Valid options are None and 'fdr_bh',
        by default 'fdr_bh'.

    Returns
    -------
    pandas.DataFrame
        Table with each label and the following values associated with each
        label: ‘pForward’, ‘zForward’, ‘likelihoodForward’,‘pReverse’,
        ‘zReverse’, and ‘probReverse’.
    """

    if data_dir is None:
        data_dir = os.path.join(str(Path.home()), "nimare_data")

    mni152 = load_mni152_brain_mask()

    stat_image = tempfile.NamedTemporaryFile(suffix=".nii.gz")
    mask_image = tempfile.NamedTemporaryFile(suffix=".nii.gz")

    multi_surface_to_volume(
        pial=pial,
        white=white,
        volume_template=mni152,
        output_file=stat_image.name,
        labels=stat_labels,
        interpolation=interpolation,
    )
    multi_surface_to_volume(
        pial=pial,
        white=white,
        volume_template=mni152,
        output_file=mask_image.name,
        labels=mask_labels,
        interpolation=interpolation,
    )

    dataset = fetch_nimare_dataset(data_dir, mask=mask_image.name, keep_neurosynth=True)

    logging.info(
        "If you use BrainStat's surface decoder, "
        + "please cite NiMARE (https://zenodo.org/record/4408504#.YBBPAZNKjzU))."
    )

    decoder = nimare.decode.continuous.CorrelationDecoder(
        feature_group=feature_group, features=features
    )
    decoder.fit(dataset)
    return decoder.transform(stat_image.name)


def fetch_nimare_dataset(
    data_dir: str,
    mask: Optional[Union[str, NiftiImage, NiftiMasker]] = None,
    keep_neurosynth: bool = True,
) -> str:
    """Downloads the nimare dataset and fetches its path.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the dataset will be saved.
    mask : str, nibabel.nifti1.Nifti1Image, nilearn.input_data.NiftiMasker or similar, or None, optional
        Mask(er) to use. If None, uses the target space image, with all non-zero
        voxels included in the mask.
    keep_neurosynth : bool, optional
        If true, then the neurosynth data files are kept, by default False.
        Note that this will not delete existing neurosynth files.

    Returns
    -------
    str
        Path to the nimare dataset.
    """

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    neurosynth_exist = os.path.isfile(os.path.join(data_dir, "database.txt"))
    if keep_neurosynth or neurosynth_exist:
        ns_dir = data_dir
    else:
        D = tempfile.TemporaryDirectory()
        ns_dir = D.name

    ns_data_file, ns_feature_file = fetch_neurosynth_dataset(ns_dir, return_pkl=False)  # type: ignore

    ns_dict = nimare.io.convert_neurosynth_to_dict(
        ns_data_file, annotations_file=ns_feature_file
    )
    dset = nimare.dataset.Dataset(ns_dict, mask=mask)
    dset = nimare.extract.download_abstracts(dset, "tsalo006@fiu.edu")
    dset.update_path(data_dir)

    return dset


def fetch_neurosynth_dataset(
    data_dir: str, return_pkl: bool = True
) -> Union[Tuple[str, str], str]:
    """Downloads the Neurosynth dataset

    Parameters
    ----------
    data_dir : str
        Directory in which to download the dataset.
    return_pkl : bool
        If true, creates and returns the .pkl file. Otherwise returns
        the dataset and features files.

    Returns
    -------
    tuple, str
        If save_pkl is false, returns a tuple containing the path to the
        database.txt and the features.txt file. Otherwise returns the path
        to the .pkl file.

    """
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    dataset_file = os.path.join(data_dir, "database.txt")
    if not os.path.isfile(dataset_file):
        logging.info("Downloading the Neurosynth dataset.")
        download(data_dir, unpack=True)
    feature_file = os.path.join(data_dir, "features.txt")

    if return_pkl:
        pkl_file = os.path.join(data_dir, "dataset.pkl")
        if not os.path.isfile(pkl_file):
            logging.info(
                "Converting Neurosynth data to a .pkl file. This may take a while."
            )
            dataset = Dataset(dataset_file, feature_file)
            dataset.save(pkl_file)
        return pkl_file

    return (dataset_file, feature_file)
