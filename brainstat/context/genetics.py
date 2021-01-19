
"""Genetic decoding using abagen."""

import numpy as np
import nibabel as nib
from abagen import get_expression_data
import tempfile
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from brainstat.mesh.interpolation import surface_to_volume
from .utils import load_mesh_labels, combine_parcellations, read_surface_gz


def surface_genetic_expression(
                        pial,
                        white,
                        labels,
                        volume_template,
                        *,
                        atlas_info=None,
                        ibf_threshold=0.5,
                        probe_selection='diff_stability',
                        donor_probes='aggregate',
                        lr_mirror=False,
                        exact=True,
                        tolerance=2,
                        sample_norm='srs',
                        gene_norm='srs',
                        norm_matched=True,
                        region_agg='donors',
                        agg_metric='mean',
                        corrected_mni=True,
                        reannotated=True,
                        return_counts=False,
                        return_donors=False,
                        donors='all',
                        data_dir=None,
                        verbose=1,
                        n_proc=1):
    """Computes genetic expression of surface parcels.

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
    volume_template : str, nibabel.nifti1.Nifti1Image
        Path to a nifti file to use as a template for the surface to volume
        procedure, or a loaded NIfTI image.

    For details of the remaining parameters please consult the
    abagen.get_expression_data() documentation. All its parameters bar "atlas"
    are valid input parameters.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the expression of each gene within each region.

    Notes
    -----
    An equal number of pial/white surfaces and labels must be provided. If
    parcellations overlap across surfaces, then the labels are kept for the
    first provided surface.

    Examples
    --------
    >>> from brainstat.context.genetics import surface_genetic_expression
    >>> from nilearn import datasets

    >>> fsaverage = datasets.fetch_surf_fsaverage()
    >>> destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    >>> parcellation = destrieux_atlas['map_left']
    >>> mni152 = datasets.load_mni152_template()
    >>> surface_genetic_expression(fsaverage['pial_left'], fsaverage['white_left'],
    ...    parcellation, mni152)
    """

    # Deal with variety of ways to provide input.
    if type(pial) is not type(white):
        ValueError('Pial and white must be of the same type.')

    if not isinstance(pial, list):
        pial = [pial]
        white = [white]

    if not isinstance(labels, list):
        labels = [labels]

    if len(pial) is not len(white):
        ValueError('The same number of pial and white surfces must be provided.')

    for i in range(len(pial)):
        if not isinstance(pial[i], BSPolyData):
            pial[i] = read_surface_gz(pial[i])

        if not isinstance(white[i], BSPolyData):
            white[i] = read_surface_gz(white[i])

    if not isinstance(volume_template, nib.nifti1.Nifti1Image):
        volume_template = nib.load(volume_template)

    for i in range(len(labels)):
        if not isinstance(labels[i],np.ndarray):
            labels[i] = load_mesh_labels(labels[i])

    # Surface data to volume.
    T = []
    for i in range(len(pial)):
        T.append(tempfile.NamedTemporaryFile(suffix='.nii.gz'))
        surface_to_volume(pial[i], white[i], labels[i], volume_template, T[i].name, verbose=verbose>0)
    if len(T) > 1:
        P = tempfile.TemporaryFile(suffix='.nii.gz')
        P_name = P.name
        T_names = [x.name for x in T]
        combine_parcellations(T_names,P.name)
    else:
        P_name = T[0].name

    # Use abagen to grab expression data.
    print("If you use BrainStat's genetics functionality, please cite abagen (https://abagen.readthedocs.io/en/stable/citing.html).")
    import pdb
    pdb.set_trace()
    expression = get_expression_data(
                               P_name,
                               atlas_info=atlas_info,
                               ibf_threshold=ibf_threshold,
                               probe_selection=probe_selection,
                               donor_probes=donor_probes,
                               lr_mirror=lr_mirror,
                               exact=exact,
                               tolerance=tolerance,
                               sample_norm=sample_norm,
                               gene_norm=gene_norm,
                               norm_matched=norm_matched,
                               region_agg=region_agg,
                               agg_metric=agg_metric,
                               corrected_mni=corrected_mni,
                               reannotated=reannotated,
                               return_counts=return_counts,
                               return_donors=return_donors,
                               donors=donors,
                               data_dir=data_dir,
                               verbose=verbose,
                               n_proc=n_proc)

    return expression