""" Meta-analytic decoding based on NiMARE """

import os
import tempfile
import nibabel as nib
from pathlib import Path
from nilearn import datasets as nil_datasets

from neurosynth.base.dataset import download, Dataset
from neurosynth import decode

import nimare
from nimare.decode import discrete
from .utils import mutli_surface_to_volume


def surface_decode_neurosynth(
        pial,
        white,
        labels,
        interpolation='linear',
        data_dir=None,
        volume_template=None,
        features=None,
        verbose=True):
    """Decodes surface data with Neurosynth

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
    interpolation : str, optional
        Either 'nearest' for nearest neighbor interpolation, or 'linear'
        for trilinear interpolation, by default 'linear'.
    data_dir : str, optional
        The directory of the nimare dataset. If none exists, a new dataset will
        be downloaded and saved to this path. If None, the directory defaults to
        your home directory, by default None.
    volume_template : str, nibabel.nifti1.Nifti1Image
        Filename of a nifti image in MNI152 space or a NIfTI image loaded with nibabel.
    features : list
        List of strings containing the names of features to include in the decoder. 
    verbose : bool, optional
        If true prints additional output to the console, by default True.

    Returns
    -------
    [type]
        [description]
    """

    # Set up the Neurosynth decoder
    if data_dir is None:
        data_dir = os.path.join(str(Path.home()), 'nimare_data')

    if volume_template is None:
        volume_template = nil_datasets.load_mni152_template()

    dataset_file = fetch_neurosynth_dataset(data_dir, return_pkl=True, verbose=verbose)
    dataset = Dataset.load(dataset_file)
    decoder = decode.Decoder(dataset, features=features)

    # Interpolate surface data to the volume and decode.
    with tempfile.NamedTemporaryFile(suffix='.nii.gz') as F:
        mutli_surface_to_volume(pial, white, volume_template,
                                labels, F.name, verbose=verbose, interpolation=interpolation)
        return decoder.decode(F)


def surface_decode_nimare(
        pial,
        white,
        labels,
        threshold=0,
        interpolation='linear',
        decoder='neurosynth',
        data_dir=None,
        verbose=True,
        correction='fdr_bh'):
    """Meta-analytic decoding of surface maps using NeuroSynth or Brainmap.

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
    threshold : float, int, optional
        Value at which to threshold the labels in volume space. Voxels equal to
        or below this value are set to 0, defaults to 0.
    interpolation : str, optional
        Either 'nearest' for nearest neighbor interpolation, or 'linear'
        for trilinear interpolation, by default 'linear'.
    decoder : str, optional
        Either 'neurosynth' for the neurosynth decoder or 'brainmap' for the
        brainmap decoder, by default 'neurosynth'.
    data_dir : str, optional
        The directory of the nimare dataset. If none exists, a new dataset will
        be downloaded and saved to this path. If None, the directory defaults to
        your home directory, by default None.
    verbose : bool, optional
        If true prints additional output to the console, by default True.
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
        data_dir = os.path.join(str(Path.home()), 'nimare_data')

    dset = fetch_nimare_dataset(data_dir)

    F = tempfile.NamedTemporaryFile(suffix='.nii.gz')
    mutli_surface_to_volume(pial, white, dset.masker.mask_img,
                            labels, F.name, verbose=verbose, interpolation=interpolation)
    nii = nib.load(F.name)
    nii2 = nib.Nifti1Image(
        (nii.get_fdata() > threshold).astype(float), nii.affine)
    ids = dset.get_studies_by_mask(nii2)

    print("If you use BrainStat's surface decoder, please cite NiMARE (https://zenodo.org/record/4408504#.YBBPAZNKjzU)).")
    if decoder is 'neurosynth':
        decoder = discrete.NeurosynthDecoder(correction=correction)
    elif decoder is 'brainmap':
        decoder = discrete.BrainMapDecoder(correction=correction)

    decoder.fit(dset)
    return decoder.transform(ids=ids)


def fetch_nimare_dataset(data_dir, keep_neurosynth=False):
    """Downloads the nimare dataset and fetches its path.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the dataset will be saved.
    keep_neurosynth : bool, optional
        If true, then the neurosynth data files are kept, by default False.

    Returns
    -------
    nimare.Dataset
        Downloaded NiMARE dataset.
    """

    nimare_file = os.path.join(
        data_dir, "neurosynth_nimare_with_abstracts.pkl.gz")
    if os.path.isfile(nimare_file):
        dset = nimare.dataset.Dataset.load(os.path.join(
            data_dir, "neurosynth_nimare_with_abstracts.pkl.gz"))
        return dset

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if keep_neurosynth:
        ns_dir = data_dir
    else:
        D = tempfile.TemporaryDirectory()
        ns_dir = D.name

    ns_data_file, ns_feature_file = fetch_neurosynth_dataset(ns_dir)

    dset = nimare.io.convert_neurosynth_to_dataset(
        ns_data_file, ns_feature_file)
    dset = nimare.extract.download_abstracts(dset, "tsalo006@fiu.edu")
    dset.save(os.path.join(data_dir, "neurosynth_nimare_with_abstracts.pkl.gz"))

    if not keep_neurosynth:
        D.cleanup()

    return dset


def fetch_neurosynth_dataset(data_dir, return_pkl=True, verbose=False):
    """Downloads the Neurosynth dataset

    Parameters
    ----------
    data_dir : str
        Directory in which to download the dataset.
    return_pkl : bool
        If true, creates and returns the .pkl file. Otherwise returns
        the dataset and features files.
    verbose : bool
        If true prints additional output to the console, by default False.

    Returns
    -------
    tuple, str
        If save_pkl is false, returns a tuple containing the path to the
        database.txt and the features.txt file. Otherwise returns the path
        to the .pkl file. 

    """
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    dataset_file = os.path.join(data_dir, 'database.txt')
    if not os.path.isfile(dataset_file):
        if verbose:
            print("Downloading the Neurosynth dataset.")
        download(data_dir, unpack=True)
    feature_file = os.path.join(data_dir, 'features.txt')

    if return_pkl:
        pkl_file = os.path.join(data_dir, 'dataset.pkl')
        if not os.path.isfile(pkl_file):
            print("Converting Neurosynth data to a .pkl file. This may take a while.")
            dataset = Dataset(dataset_file, feature_file)
            dataset.save(pkl_file)
        return pkl_file

    return (dataset_file, feature_file)
