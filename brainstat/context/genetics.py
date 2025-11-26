"""Genetic decoding using abagen."""
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union
import os
import nibabel as nib
import numpy as np
import pandas as pd

# Monkeypatch for abagen compatibility with pandas 2.0+
if not hasattr(pd.DataFrame, 'append'):
    def _append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        from pandas import concat
        if isinstance(other, (list, tuple)):
            to_concat = [self] + list(other)
        else:
            to_concat = [self, other]
        return concat(to_concat, ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)
    pd.DataFrame.append = _append

# Monkeypatch for abagen compatibility with pandas 2.0+ (set_axis inplace)
_original_set_axis = pd.DataFrame.set_axis
def _set_axis_patched(self, labels, *args, **kwargs):
    if 'inplace' in kwargs:
        del kwargs['inplace']
    return _original_set_axis(self, labels, *args, **kwargs)
pd.DataFrame.set_axis = _set_axis_patched

import abagen.images

def _labeltable_to_df_patched(labels):
    """
    Patched version of abagen.utils.labeltable_to_df to handle missing 0 index
    and use pd.concat instead of append.
    """
    info = pd.DataFrame(columns=['id', 'label', 'hemisphere', 'structure'])
    for table, hemi in zip(labels, ('L', 'R')):
        if len(table) == 0:
            continue
        ids, label = zip(*table.items())
        new_df = pd.DataFrame(dict(id=ids, label=label, hemisphere=hemi, structure='cortex'))
        info = pd.concat([info, new_df], ignore_index=True)
        
    # Use errors='ignore' to handle missing 0
    info = info.set_index('id').drop([0], axis=0, errors='ignore').sort_index()

    if len(info) != 0:
        return info

abagen.images.labeltable_to_df = _labeltable_to_df_patched


import collections
from abagen import check_atlas, get_expression_data
from brainspace.mesh.mesh_io import read_surface, write_surface
from sklearn.model_selection import ParameterGrid
from nibabel.gifti import GiftiImage, GiftiDataArray

from brainstat._utils import data_directories, logger
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
    atlas_info: Optional[str] = None,
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
        List of paths to surface files or preloaded surfaces. If not specified
        assumes that `labels` are on the `fsaverage5` surface. Default: None
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

    # Deal with the input parameters.
    if isinstance(surfaces, str):
        surfaces = [surfaces]
    elif surfaces is None:
        surfaces = []

    temp_files = []
    try:
        if isinstance(labels, np.ndarray):
            # Assuming 'labels' is a 1D NumPy array of length 20484
            num_vertices = len(labels)  # Should be 20484
            half_size = num_vertices // 2  # Half of 20484, which is 10242

            # Split the array into two halves
            labels_left = labels[:half_size]  # First half for the left hemisphere
            labels_right = labels[half_size:]  # Second half for the right hemisphere

            # Create GiftiDataArrays for each hemisphere
            data_array_left = GiftiDataArray(data=labels_left)
            data_array_right = GiftiDataArray(data=labels_right)

            # Create separate GiftiImages for each hemisphere
            labels_left_gii = GiftiImage(darrays=[data_array_left])
            labels_right_gii = GiftiImage(darrays=[data_array_right])
            
            # Save to temporary files for abagen compatibility
            labels_files = []
            for img in [labels_left_gii, labels_right_gii]:
                f = tempfile.NamedTemporaryFile(suffix=".gii", delete=False)
                f.close()
                nib.save(img, f.name)
                labels_files.append(f.name)
                temp_files.append(f.name)
            labels = tuple(labels_files)
    
        surfaces_gii = []
        for surface in surfaces:
            if not isinstance(surface, str) and not isinstance(surface, Path):
                # Cast surface data to float32 to comply with GIFTI standard
                # GIFTI only supports uint8, int32, and float32 datatypes
                if hasattr(surface, 'Points') and surface.Points.dtype != np.float32:
                    surface = surface.copy()
                    surface.Points = surface.Points.astype(np.float32)
                
                f = tempfile.NamedTemporaryFile(suffix=".gii", delete=False)
                f.close()
                write_surface(surface, f.name, otype="gii")
                surfaces_gii.append(f.name)
                temp_files.append(f.name)
            else:
                surfaces_gii.append(surface)

        # Use abagen to grab expression data.
        logger.info(
            "If you use BrainStat's genetics functionality, please cite abagen (https://abagen.readthedocs.io/en/stable/citing.html)."
        )
        atlas = check_atlas(labels, geometry=surfaces_gii, space=space)
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
    finally:
        for f in temp_files:
            try:
                Path(f).unlink()
            except FileNotFoundError:
                pass


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
        # We'll use fsaverage5 for all.
        parcellations[key].update({"surfaces": ["fsaverage5"]})

    param_grid = ParameterGrid(list(parcellations.values()))

    # Compute expression for all parcellations.
    for params in param_grid:
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
        output_file = Path(output_dir) / filename
        if output_file.exists():
            print('Skipping "{}"'.format(output_file))
            continue
        else:
            print('Computing "{}"'.format(output_file))

        surface_files = _fetch_template_surface_files(
            params["surfaces"], data_dir=data_dir  # type: ignore
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
        expression.to_csv(output_file)  # type: ignore


def __freesurfer_to_surfgii(freesurfer_file: str, gifti_file: str) -> None:
    surf = read_surface(freesurfer_file, itype="fs")
    write_surface(surf, gifti_file, otype="gii")
