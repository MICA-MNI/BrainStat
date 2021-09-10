""" Load external datasets. """
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from brainspace.mesh.mesh_creation import build_polydata
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_operations import combine_surfaces
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from netneurotools import datasets as nnt_datasets
from netneurotools.civet import read_civet
from nibabel import load as nib_load
from nibabel.freesurfer.io import read_annot, read_geometry

from brainstat._utils import _download_file, data_directories, read_data_fetcher_json


def fetch_parcellation(
    template: str,
    atlas: str,
    n_regions: int,
    join: bool = True,
    seven_networks: bool = True,
    data_dir: Optional[Union[str, Path]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Loads the surface parcellation of a given atlas.

    Parameters
    ----------
    template : str,
        The surface template. Valid values are "fsaverage", "fsaverage5",
        "fsaverage6", "fslr32k", by default "fsaverage5".
    atlas : str
        Name of the atlas. Valid names are "schaefer", "cammoun", "glasser".
    n_regions : int
        Number of regions of the requested atlas. Valid values for the "schaefer" atlas are
        100, 200, 300, 400, 500, 600, 800, 1000. Valid values for the cammoun atlas are 33,
        60, 125, 250, 500. Valid values for the glasser atlas are 360.
    join : bool, optional
        If true, returns parcellation as a single array, if false, returns an
        array per hemisphere, by default True.
    seven_networks : bool, optional
        If true, uses the 7 networks parcellation. Only used for the Schaefer
        atlas, by default True.
    data_dir : str, pathlib.Path, optional
        Directory to save the data, defaults to $HOME_DIR/brainstat_data/parcellation_data.

    Returns
    -------
    np.ndarray or tuple of np.npdarray
        Surface parcellation. If a tuple, then the first element is the left hemisphere.
    """

    data_dir = Path(data_dir) if data_dir else data_directories["PARCELLATION_DATA_DIR"]
    data_dir.mkdir(parents=True, exist_ok=True)

    if atlas == "schaefer":
        parcellations = _fetch_schaefer_parcellation(
            template, n_regions, seven_networks, data_dir
        )
    elif atlas == "cammoun":
        parcellations = _fetch_cammoun_parcellation(template, n_regions, data_dir)
    elif atlas == "glasser":
        parcellations = _fetch_glasser_parcellation(template, data_dir)
    else:
        raise ValueError(f"Invalid atlas: {atlas}")

    if join:
        return np.concatenate((parcellations[0], parcellations[1]), axis=0)
    else:
        return parcellations[0], parcellations[1]


def fetch_template_surface(
    template: str,
    join: bool = True,
    layer: Optional[str] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> Union[BSPolyData, Tuple[BSPolyData, BSPolyData]]:
    """Loads surface templates.

    Parameters
    ----------
    template : str
        Name of the surface template. Valid values are "fslr32k", "fsaverage",
        "fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "civet41k",
        "civet164k".
    join : bool, optional
        If true, returns surfaces as a single object, if false, returns an object per hemisphere, by default True.
    layer : str, optional
        Name of the cortical surface of interest. Valid values are "white",
        "smoothwm", "pial", "inflated", "sphere" for fsaverage surfaces and
        "midthickness", "inflated", "vinflated" for "fslr32k". If None,
        defaults to "pial" or "midthickness", by default None.
    data_dir : str, Path, optional
        Directory to save the data, by default $HOME_DIR/brainstat_data/surface_data.

    Returns
    -------
    BSPolyData or tuple of BSPolyData
        Output surface(s). If a tuple, then the first element is the left hemisphere.
    """

    data_dir = Path(data_dir) if data_dir else data_directories["SURFACE_DATA_DIR"]
    surface_files = _fetch_template_surface_files(template, layer, data_dir)
    if template[:9] == "fsaverage":
        surfaces_fs = [read_geometry(file) for file in surface_files]
        surfaces = [build_polydata(surface[0], surface[1]) for surface in surfaces_fs]
    elif template == "fslr32k":
        surfaces = [read_surface(file) for file in surface_files]
    else:
        surfaces_obj = [read_civet(file) for file in surface_files]
        surfaces = [build_polydata(surface[0], surface[1]) for surface in surfaces_obj]

    if join:
        return combine_surfaces(surfaces[0], surfaces[1])
    else:
        return surfaces[0], surfaces[1]


def fetch_mask(
    template: str,
    join: bool = True,
    data_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Fetches midline masks.

    Parameters
    ----------
    template : str
        Name of the surface template. Valid templates are "civet41k" and "civet164k".
    join : bool, optional
        If true, returns a numpy array containing the mask. If false, returns a
        tuple containing the left and right hemispheric masks, respectively, by
        default True
    data_dir : str, pathlib.Path, optional
        Directory to save the data, by default $HOME_DIR/brainstat_data/surface_data.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Midline mask, either as a single array or a tuple of a left and right hemispheric
        array.
    """
    data_dir = Path(data_dir) if data_dir else data_directories["SURFACE_DATA_DIR"]
    mask_file = data_dir / f"{template}_mask.csv"
    url = read_data_fetcher_json()["masks"][template]["url"]
    _download_file(url, mask_file, overwrite=overwrite)

    mask = np.loadtxt(mask_file, delimiter=",") == 1
    if join:
        return mask
    else:
        n = len(mask)
        return mask[: n // 2], mask[n // 2 :]


def _fetch_template_surface_files(
    template: str,
    layer: Optional[str] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> Tuple[str, str]:
    """Fetches surface files.

    Parameters
    ----------
    template : str
        Name of the surface template. Valid values are "fslr32k", "fsaverage",
        "fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "civet41k",
        "civet164k".
      layer : str, optional
        Name of the cortical surface of interest. Valid values are "white",
        "smoothwm", "pial", "inflated", "sphere" for fsaverage surfaces,
        "midthickness", "inflated", "vinflated" for "fslr32k", and "mid,
        "white" for civet surfaces. If None, defaults to "pial", "midthickness",
        or "mid", by default None.
    data_dir : str, optional
        Directory to save the data, by default None.

    Returns
    -------
    Tuple of str
        Surface files.
    """

    if template == "fslr32k":
        layer = layer if layer else "midthickness"
        bunch = nnt_datasets.fetch_conte69(data_dir=str(data_dir))
    elif template == "civet41k" or template == "civet164k":
        layer = layer if layer else "mid"
        bunch = nnt_datasets.fetch_civet(
            density=template[5:], version="v2", data_dir=str(data_dir)
        )
    else:
        layer = layer if layer else "pial"
        bunch = nnt_datasets.fetch_fsaverage(version=template, data_dir=str(data_dir))
    return bunch[layer]


def _valid_parcellations() -> dict:
    """Returns a dictionary of valid parcellations."""
    return {
        "schaefer": {
            "n_regions": (100, 200, 300, 400, 500, 600, 800, 1000),
            "surfaces": ("fsaverage5", "fsaverage6", "fsaverage", "fslr32k"),
        },
        "cammoun": {
            "n_regions": (33, 60, 125, 250, 500),
            "surfaces": ("fsaverage5", "fsaverage6", "fsaverage", "fslr32k"),
        },
        "glasser": {
            "n_regions": (360,),
            "surfaces": ("fsaverage5", "fsaverage", "fslr32k"),
        },
    }


def _fetch_schaefer_parcellation(
    template: str, n_regions: int, seven_networks: int, data_dir: Path
) -> List[np.ndarray]:
    """Fetches Schaefer parcellations."""
    n_networks = 7 if seven_networks else 17
    key = f"{n_regions}Parcels{n_networks}Networks"
    bunch = nnt_datasets.fetch_schaefer2018(version=template, data_dir=str(data_dir))
    if template == "fslr32k":
        cifti = nib_load(bunch[key])
        parcellation_full = np.squeeze(cifti.get_fdata())
        parcellations = [x for x in np.reshape(parcellation_full, (2, -1))]
    else:
        parcellations = [read_annot(file)[0] for file in bunch[key]]
        parcellations[1] += n_regions // 2
    return parcellations


def _fetch_cammoun_parcellation(
    template: str, n_regions: int, data_dir: Path
) -> List[np.ndarray]:
    """Fetches Cammoun parcellations."""
    key = f"scale{n_regions:03}"
    bunch = nnt_datasets.fetch_cammoun2012(version=template, data_dir=str(data_dir))
    if template == "fslr32k":
        gifti = [nib_load(file) for file in bunch[key]]
        parcellations = [x.darrays[0].data for x in gifti]
    else:
        parcellations = [read_annot(file)[0] for file in bunch[key]]
    return parcellations


def _fetch_glasser_parcellation(template: str, data_dir: Path) -> List[np.ndarray]:
    """Fetches Glasser parcellation."""
    urls = read_data_fetcher_json()["parcellations"]["glasser"][template]["url"]
    filepaths = []
    for i, hemi in enumerate(("lh", "rh")):
        filename = "_".join(("glasser", "360", template, hemi)) + "label.gii"
        filepaths.append(data_dir / filename)
        _download_file(urls[i], filepaths[i])
    gifti = [nib_load(file) for file in filepaths]
    parcellations = [x.darrays[0].data for x in gifti]
    parcellations[1] = (parcellations[1] + 180) * (parcellations[1] > 0)
    return parcellations
