"""Genetic decoding using abagen."""
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from abagen import check_atlas, get_expression_data
from brainspace.mesh.mesh_io import read_surface, write_surface
from sklearn.model_selection import ParameterGrid

from brainstat._utils import data_directories
from brainstat.datasets.base import (
    _fetch_template_surface_files,
    _valid_parcellations,
    fetch_parcellation,
)


def surface_genetic_expression(
    labels: Union[Sequence[str], np.ndarray],
    surfaces: Optional[Union[str, Sequence[str]]] = None,
    space: Optional[str] = None,
    *,
    atlas_info: str = None,
    ibf_threshold: float = 0.5,
    probe_selection: str = "diff_stability",
    donor_probes: str = "aggregate",
    lr_mirror: Optional[bool] = None,
    missing: Optional[str] = None,
    tolerance: float = 2,
    sample_norm: str = "srs",
    gene_norm: str = "srs",
    norm_matched: bool = True,
    norm_structures: bool = False,
    region_agg: str = "donors",
    agg_metric: str = "mean",
    corrected_mni: bool = True,
    reannotated: bool = True,
    return_counts: bool = False,
    return_donors: bool = False,
    return_report: bool = False,
    donors: str = "all",
    data_dir: Optional[str] = None,
    verbose: float = 0,
    n_proc: int = 1
) -> pd.DataFrame:
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


def __create_precomputed(
    data_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Creates precomputed matrices of genetic expression for standard atlases. These are
    used for the MATLAB implementation."""

    if output_dir is None:
        output_dir = data_dir = (
            Path(data_dir) if data_dir else data_directories["BRAINSTAT_DATA_DIR"]
        )

    # Get all parcellations in a format compatible with ParameterGrid.
    parcellations = _valid_parcellations()
    for key in parcellations:
        parcellations[key].update({"atlas": [key]})
        if key == "schaefer":
            parcellations[key].update({"seven_networks": (True, False)})
        else:
            parcellations[key].update({"seven_networks": (True,)})

        # We only really need to compute this for one template surface.
        # We'll use fsaverage for all.
        parcellations[key].update({"surfaces": ["fsaverage"]})

    param_grid = ParameterGrid(list(parcellations.values()))

    # Compute expression for all parcellations.
    for params in param_grid:
        surface_files = _fetch_template_surface_files(
            params["surfaces"], data_dir=data_dir
        )
        space = "fslr" if params["surfaces"] == "fslr32k" else "fsaverage"

        labels = fetch_parcellation(
            params["surfaces"],
            params["atlas"],
            params["n_regions"],
            seven_networks=params["seven_networks"],
            data_dir=data_dir,
        )

        surf_lh = tempfile.NamedTemporaryFile(suffix=".surf.gii")
        surf_rh = tempfile.NamedTemporaryFile(suffix=".surf.gii")
        for i, surf in enumerate((surf_lh, surf_rh)):
            __freesurfer_to_surfgii(surface_files[i], surf.name)
        surface_paths = [surf_lh.name, surf_rh.name]

        expression = surface_genetic_expression(labels, surface_paths, space=space)  # type: ignore

        if params["atlas"] == "schaefer":
            network_tag = "7Networks" if params["seven_networks"] else "17Networks"
        else:
            network_tag = ""

        filename_components = filter(
            None,
            (
                "expression",
                params["atlas"],
                str(params["n_regions"]),
                network_tag,
            ),
        )
        filename = "_".join(filename_components) + ".csv.gz"
        expression.to_csv(Path(output_dir, filename))  # type: ignore


def __freesurfer_to_surfgii(freesurfer_file: str, gifti_file: str) -> None:
    surf = read_surface(freesurfer_file, itype="fs")
    write_surface(surf, gifti_file, otype="gii")
