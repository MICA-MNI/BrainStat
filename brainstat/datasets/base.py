""" Load external datasets. """
from typing import Optional, Tuple, Union

import numpy as np
from brainspace.mesh.mesh_creation import build_polydata
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_operations import combine_surfaces
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from netneurotools import datasets as nnt_datasets
from nibabel import load as nib_load
from nibabel.freesurfer.io import read_annot, read_geometry


def fetch_parcellation(
    atlas: str,
    n_regions: Union[int, str],
    template: str = "fsaverage5",
    join: bool = True,
    seven_networks: bool = True,
    data_dir: Optional[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Loads the surface parcellation of a given atlas.

    Parameters
    ----------
    atlas : str
        Name of the atlas. Valid names are "schaefer", "cammoun".
    n_regions : int, str
        Number of regions of the requested atlas. Valid values for the "schaefer " atlas are
        100, 200, 300, 400, 500, 600, 800, 1000. Valid values for the cammoun atlas are 33,
        60, 125, 250, 500.
    template : str, optional
        The surface template. Valid values are "fsaverage", "fsaverage5",
        "fsaverage6", "fslr32k", by default "fsaverage5".
    join : bool, optional
        If true, returns parcellation as a single array, if false, returns an
        array per hemisphere, by default True.
    seven_networks : bool, optional
        If true, uses the 7 networks parcellation. Only used for the Schaefer
        atlas, by default True.
    data_dir : str, optional
        Directory to save the data, by default None.

    Returns
    -------
    np.ndarray or tuple of np.npdarray
        Surface parcellation. If a tuple, then the first element is the left hemisphere.
    """

    if atlas == "schaefer":
        n_networks = 7 if seven_networks else 17
        key = f"{n_regions}Parcels{n_networks}Networks"
        bunch = nnt_datasets.fetch_schaefer2018(version=template, data_dir=data_dir)
        if template == "fslr32k":
            cifti = nib_load(bunch[key])
            parcellation_full = np.squeeze(cifti.get_fdata())
            parcellations = [x for x in np.reshape(parcellation_full, (2, -1))]
        else:
            parcellations = [read_annot(file)[0] for file in bunch[key]]
            parcellations[1] += n_regions // 2

    elif atlas == "cammoun":
        key = f"scale{n_regions:03}"
        bunch = nnt_datasets.fetch_cammoun2012(version=template, data_dir=data_dir)
        if template == "fslr32k":
            gifti = [nib_load(file) for file in bunch[key]]
            parcellations = [x.darrays[0].data for x in gifti]
        else:
            parcellations = [read_annot(file)[0] for file in bunch[key]]

    if join:
        return np.concatenate((parcellations[0], parcellations[1]), axis=0)
    else:
        return parcellations[0], parcellations[1]


def fetch_template_surface(
    template: str,
    join: bool = True,
    layer: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Union[BSPolyData, Tuple[BSPolyData, BSPolyData]]:
    """Loads surface templates.

    Parameters
    ----------
    template : str
        Name of the surface template. Valid values are "fslr32k", "fsaverage",
        "fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6".
    join : bool, optional
        If true, returns surfaces as a single object, if false, returns an object per hemisphere, by default True.
    layer : str, optional
        Name of the cortical surface of interest. Valid values are "white",
        "smoothwm", "pial", "inflated", "sphere" for fsaverage surfaces and
        "midthickness", "inflated", "vinflated" for "fslr32k". If None,
        defaults to "pial" or "midthickness", by default None.
    data_dir : str, optional
        Directory to save the data, by default None.

    Returns
    -------
    BSPolyData or tuple of BSPolyData
        Output surface(s). If a tuple, then the first element is the left hemisphere.
    """

    if template == "fslr32k":
        layer = layer if layer else "midthickness"
        bunch = nnt_datasets.fetch_conte69(data_dir=data_dir)
        surfaces = [read_surface(file) for file in bunch[layer]]
    else:
        layer = layer if layer else "pial"
        bunch = nnt_datasets.fetch_fsaverage(data_dir=data_dir)
        surfaces_fs = [read_geometry(file) for file in bunch[layer]]
        surfaces = [build_polydata(surface[0], surface[1]) for surface in surfaces_fs]

    if join:
        return combine_surfaces(surfaces[0], surfaces[1])
    else:
        return surfaces[0], surfaces[1]
