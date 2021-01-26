""" Meta-analytic decoding based on NiMARE """

import os
import tempfile
import nibabel as nib
from pathlib import Path

from neurosynth.base.dataset import download
import nimare
from nimare.decode import discrete
from .utils import mutli_surface_to_volume


def surface_decode(
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
        your home directory, by default None
    verbose : bool, optional
        If true prints additional output to the console, by default True
    correction : str, optional
        Multiple comparison correction. Valid options are None and 'fdr_bh',
        by default 'fdr_bh'

    Returns
    -------
    pandas.DataFrame
        Table with each label and the following values associated with each
        label: ‘pForward’, ‘zForward’, ‘likelihoodForward’,‘pReverse’,
        ‘zReverse’, and ‘probReverse’.
    """

    if data_dir is None:
        data_dir = os.path.join(str(Path.home()), 'nimare_data')

    dset = fetch_dataset(data_dir)

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


def fetch_dataset(data_dir, keep_neurosynth=False):
    """Downloads the nimare dataset and fetches its path.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the dataset will be saved.
    keep_neurosynth : bool, optional
        If true, then the neurosynth data files are kept, by default False.

    Returns
    -------
    str
        Path to the nimare dataset.
    """

    if os.path.isfile(os.path.join(data_dir, "neurosynth_nimare_with_abstracts.pkl.gz")):
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

    download(ns_dir, unpack=True)

    dset = nimare.io.convert_neurosynth_to_dataset(
        os.path.join(ns_dir, "database.txt"), os.path.join(ns_dir, "features.txt"))
    dset = nimare.extract.download_abstracts(dset, "tsalo006@fiu.edu")
    dset.save(os.path.join(data_dir, "neurosynth_nimare_with_abstracts.pkl.gz"))

    if not keep_neurosynth:
        D.cleanup()

    return dset
