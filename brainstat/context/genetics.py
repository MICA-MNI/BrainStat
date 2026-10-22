"""Genetic decoding using abagen."""

import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Optional, Sequence, Union

import abagen.images
import nibabel as nib
import numpy as np
import pandas as pd
from abagen import check_atlas, get_expression_data
from brainspace.mesh.mesh_io import read_surface, write_surface
from nibabel.gifti import GiftiDataArray, GiftiImage
from sklearn.model_selection import ParameterGrid

from brainstat._utils import data_directories, logger
from brainstat.datasets.base import (
    _fetch_template_surface_files,
    _valid_parcellations,
    fetch_parcellation,
)

_COMPAT_LOCK = RLock()


def _labeltable_to_df(labels):
    rows = [
        dict(id=key, label=value, hemisphere=hemi, structure="cortex")
        for table, hemi in zip(labels, ("L", "R"))
        for key, value in table.items()
        if key != 0
    ]
    if rows:
        return pd.DataFrame(rows).set_index("id").sort_index()
    return None


@contextmanager
def _abagen_compat():
    """Scope legacy abagen adapters to a call and always restore originals.

    The lock serializes BrainStat genetics calls. The pandas adapters are
    process-wide while the call runs; callers using abagen/pandas directly in
    other threads should avoid running them concurrently with this workflow.
    Dependency bounds remain in place until upstream supports newer pandas.
    """
    with _COMPAT_LOCK:
        original_append = getattr(pd.DataFrame, "append", None)
        original_set_axis = pd.DataFrame.set_axis
        original_labels = abagen.images.labeltable_to_df

        def append(
            frame, other, ignore_index=False, verify_integrity=False, sort=False
        ):
            if isinstance(other, pd.Series):
                other = other.to_frame().T
            objects = [frame] + (
                list(other) if isinstance(other, (list, tuple)) else [other]
            )
            return pd.concat(
                objects,
                ignore_index=ignore_index,
                verify_integrity=verify_integrity,
                sort=sort,
            )

        def set_axis(frame, labels, *args, **kwargs):
            inplace = kwargs.pop("inplace", False)
            if inplace:
                raise ValueError(
                    "abagen compatibility supports set_axis(inplace=False) only"
                )
            return original_set_axis(frame, labels, *args, **kwargs)

        try:
            if original_append is None:
                pd.DataFrame.append = append
            pd.DataFrame.set_axis = set_axis
            abagen.images.labeltable_to_df = _labeltable_to_df
            yield
        finally:
            if original_append is None:
                del pd.DataFrame.append
            else:
                pd.DataFrame.append = original_append
            pd.DataFrame.set_axis = original_set_axis
            abagen.images.labeltable_to_df = original_labels


def _surface_size(surface):
    if isinstance(surface, (str, Path)):
        surface = read_surface(str(surface))
    return surface.n_points


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
    n_proc: int = 1,
) -> Union[pd.DataFrame, dict, tuple]:
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
    pandas.DataFrame, dict or tuple
        Expression data returned unchanged from abagen. The return_counts,
        return_donors and return_report options can change the return type.

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

    if isinstance(surfaces, (str, Path)):
        surfaces = [surfaces]
    elif surfaces is not None:
        surfaces = list(surfaces)

    if surfaces is not None and len(surfaces) != 2:
        raise ValueError("Provide both left and right hemisphere surfaces.")

    if isinstance(labels, np.ndarray):
        if (
            labels.ndim != 1
            or labels.size == 0
            or not np.issubdtype(labels.dtype, np.number)
            or np.iscomplexobj(labels)
            or not np.all(np.isfinite(labels))
            or np.any(labels < 0)
            or np.any(labels > np.iinfo(np.int32).max)
            or not np.all(labels == np.floor(labels))
        ):
            raise ValueError(
                "labels must be a nonempty 1D array of nonnegative int32 label IDs."
            )
        if surfaces is None:
            if labels.size != 20484:
                raise ValueError(
                    "Without surfaces, labels must match fsaverage5 (20484 vertices)."
                )
            surfaces = list(
                _fetch_template_surface_files(
                    "fsaverage5",
                    data_dir=(
                        Path(data_dir)
                        if data_dir
                        else data_directories["SURFACE_DATA_DIR"]
                    ),
                )
            )
            if space is None:
                space = "fsaverage"
        sizes = [_surface_size(surface) for surface in surfaces]
        if sum(sizes) != labels.size:
            raise ValueError(
                "Label count does not match the supplied surface geometry."
            )

    # A directory owns every temporary file, including files whose save fails.
    with tempfile.TemporaryDirectory(prefix="brainstat-genetics-") as tmpdir:
        if isinstance(labels, np.ndarray):
            arrays = np.split(labels.astype(np.int32), [sizes[0]])
            label_files = []
            for index, array in enumerate(arrays):
                path = str(Path(tmpdir) / f"labels-{index}.label.gii")
                image = GiftiImage(
                    darrays=[GiftiDataArray(data=array, intent="NIFTI_INTENT_LABEL")]
                )
                nib.save(image, path)
                label_files.append(path)
            labels = tuple(label_files)
        else:
            labels = tuple(str(path) for path in labels)

        surfaces_gii = None
        if surfaces is not None:
            surfaces_gii = []
            for index, surface in enumerate(surfaces):
                if isinstance(surface, (str, Path)):
                    surfaces_gii.append(str(surface))
                else:
                    # Cast a copy: converting the surface must not alter caller geometry.
                    surface = surface.copy()
                    surface.Points = surface.Points.astype(np.float32)
                    path = str(Path(tmpdir) / f"surface-{index}.surf.gii")
                    write_surface(surface, path, otype="gii")
                    surfaces_gii.append(path)

        # Use abagen to grab expression data.
        logger.info(
            "If you use BrainStat's genetics functionality, please cite abagen (https://abagen.readthedocs.io/en/stable/citing.html)."
        )
        with _abagen_compat():
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
