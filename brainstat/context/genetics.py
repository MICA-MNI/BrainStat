"""Genetic decoding using abagen."""

from abagen import check_atlas, get_expression_data


def surface_genetic_expression(
    labels,
    surfaces=None,
    space=None,
    *,
    atlas_info=None,
    ibf_threshold=0.5,
    probe_selection="diff_stability",
    donor_probes="aggregate",
    lr_mirror=None,
    missing=None,
    tolerance=2,
    sample_norm="srs",
    gene_norm="srs",
    norm_matched=True,
    norm_structures=False,
    region_agg="donors",
    agg_metric="mean",
    corrected_mni=True,
    reannotated=True,
    return_counts=False,
    return_donors=False,
    return_report=False,
    donors="all",
    data_dir=None,
    verbose=0,
    n_proc=1
):
    """Computes genetic expression of surface parcels.

    Parameters
    ----------
    labels : list-of-str or numpy.ndarray
        List of paths to label files for the parcellation, or numpy array
        containing the pre-loaded labels
    surfaces : list-of-image, optional
        List of paths to surface files. If not specified assumes that `labels`
        are on the `fsaverage5` surface. Default: None
    space : {'fsaverage', 'fslr'}
        What template space `surfaces` are aligned to. If not specified assumes
        that `labels` are on the `fsaverage5` surface. Default: None

    For details of the remaining parameters please consult the
    abagen.get_expression_data() documentation. All its parameters bar "atlas"
    are valid input parameters.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the expression of each gene within each region.

    Examples
    --------
    >>> from brainstat.context.genetics import surface_genetic_expression
    >>> from nilearn import datasets
    >>> import numpy as np

    >>> destrieux = datasets.fetch_atlas_surf_destrieux()
    >>> labels = np.hstack((destrieux['map_left'], destrieux['map_right']))
    >>> fsaverage = datasets.fetch_surf_fsaverage()
    >>> surfaces = (fsaverage['pial_left'], fsaverage['pial_right'])
    >>> expression = surface_genetic_expression(labels, surfaces,
    ...                                         space='fsaverage')
    """

    # Use abagen to grab expression data.
    print(
        "If you use BrainStat's genetics functionality, please cite abagen (https://abagen.readthedocs.io/en/stable/citing.html)."
    )
    atlas = check_atlas(labels, geometry=surfaces, space=space)
    expression = get_expression_data(
        atlas,
        atlas_info=atlas_info,
        ibf_threshold=ibf_threshold,
        probe_selection=probe_selection,
        donor_probes=donor_probes,
        lr_mirror=lr_mirror,
        missing=missing,
        tolerance=tolerance,
        sample_norm=sample_norm,
        gene_norm=gene_norm,
        norm_matched=norm_matched,
        norm_structures=norm_structures,
        region_agg=region_agg,
        agg_metric=agg_metric,
        corrected_mni=corrected_mni,
        reannotated=reannotated,
        return_counts=return_counts,
        return_donors=return_donors,
        return_report=return_report,
        donors=donors,
        data_dir=data_dir,
        verbose=verbose,
        n_proc=n_proc,
    )

    return expression
