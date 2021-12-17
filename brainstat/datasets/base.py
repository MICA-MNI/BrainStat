""" Load external datasets. """
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
from brainspace.mesh.mesh_creation import build_polydata
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_operations import combine_surfaces
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from netneurotools import datasets as nnt_datasets
from netneurotools.civet import read_civet
from nibabel import load as nib_load
from nibabel.freesurfer.io import read_annot, read_geometry

from brainstat._utils import (
    _download_file,
    data_directories,
    logger,
    read_data_fetcher_json,
)
from brainstat.mesh.interpolate import _surf2surf


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
        "fsaverage6", "fslr32k", "civet41k", "civet164k", by default "fsaverage5".
    atlas : str
        Name of the atlas. Valid names are "cammoun", "glasser", "schaefer", "yeo".
    n_regions : int
        Number of regions of the requested atlas. Valid values for the cammoun
        atlas are 33, 60, 125, 250, 500. Valid values for the glasser atlas are
        360. Valid values for the "schaefer" atlas are 100, 200, 300, 400, 500,
        600, 800, 1000. Valid values for "yeo" are 7 and 17.
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

    if template == "civet41k" or template == "civet164k":
        logger.info(
            "CIVET parcellations were not included with the toolbox. Interpolating parcellation from the fsaverage surface with a nearest neighbor interpolation."
        )
        civet_template = template
        template = "fsaverage"
    else:
        civet_template = ""

    if atlas == "schaefer":
        parcellations = _fetch_schaefer_parcellation(
            template, n_regions, seven_networks, data_dir
        )
    elif atlas == "cammoun":
        parcellations = _fetch_cammoun_parcellation(template, n_regions, data_dir)
    elif atlas == "glasser":
        parcellations = _fetch_glasser_parcellation(template, data_dir)
    elif atlas == "yeo":
        parcellations = _fetch_yeo_parcellation(template, n_regions, data_dir)
    else:
        raise ValueError(f"Invalid atlas: {atlas}")

    if civet_template:
        fsaverage_left, fsaverage_right = fetch_template_surface(
            "fsaverage", layer="white", join=False
        )
        civet_left, civet_right = fetch_template_surface(
            civet_template, layer="white", join=False
        )

        parcellations[0] = _surf2surf(
            fsaverage_left, civet_left, parcellations[0], interpolation="nearest"
        )
        parcellations[1] = _surf2surf(
            fsaverage_right, civet_right, parcellations[1], interpolation="nearest"
        )

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
        If true, returns surfaces as a single object, if false, returns an
        object per hemisphere, by default True.
    layer : str, optional
        Name of the cortical surface of interest. Valid values are "white",
        "smoothwm", "pial", "inflated", "sphere" for fsaverage surfaces;
        "midthickness", "inflated", "vinflated" for "fslr32k"; "mid", "white"
        for CIVET surfaces; and "sphere" for "civet41k". If None,
        defaults to "pial" or "midthickness", by default None.
    data_dir : str, Path, optional
        Directory to save the data, by default
        $HOME_DIR/brainstat_data/surface_data.

    Returns
    -------
    BSPolyData or tuple of BSPolyData
        Output surface(s). If a tuple, then the first element is the left
        hemisphere.
    """

    data_dir = Path(data_dir) if data_dir else data_directories["SURFACE_DATA_DIR"]
    surface_files = _fetch_template_surface_files(template, data_dir, layer)
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
        Name of the surface template. Valid templates are: "fsaverage5",
        "fsaverage", "fslr32k", "civet41k", and "civet164k".
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
    data_dir.mkdir(parents=True, exist_ok=True)

    mask_file = data_dir / f"{template}_mask.csv"
    url = read_data_fetcher_json()["masks"][template]["url"]
    _download_file(url, mask_file, overwrite=overwrite)

    mask = np.loadtxt(mask_file, delimiter=",") == 1
    if join:
        return mask
    else:
        n = len(mask)
        return mask[: n // 2], mask[n // 2 :]


def fetch_gradients(
    template: str = "fsaverage5",
    name: str = "margulies2016",
    data_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> np.ndarray:
    """Fetch example gradients.

    Parameters
    ----------
    template : str, optional
        Name of the template surface. Valid values are "fsaverage5",
        "fsaverage", "fslr32k", defaults to "fsaverage5".
    name : str
        Name of the gradients. Valid values are "margulies2016", defaults to
        "margulies2016".
    data_dir : str, Path, optional
        Path to the directory to store the gradient data files, by
        default $HOME_DIR/brainstat_data/gradient_data.
    overwrite : bool, optional
        If true, overwrites existing files, by default False.

    Returns
    -------
    numpy.ndarray
        Vertex-by-gradient matrix.
    """
    data_dir = Path(data_dir) if data_dir else data_directories["GRADIENT_DATA_DIR"]
    data_dir.mkdir(parents=True, exist_ok=True)

    gradients_file = data_dir / f"gradients_{name}.h5"
    if not gradients_file.exists() or overwrite:
        url = read_data_fetcher_json()["gradients"][name]["url"]
        _download_file(url, gradients_file, overwrite=overwrite)

    hf = h5py.File(gradients_file, "r")
    if template == "civet41k" or template == "civet164k":
        logger.info(
            "CIVET gradients were not included with the toolbox. Interpolating gradients from the fsaverage surface with a nearest interpolation."
        )
        fsaverage_left, fsaverage_right = fetch_template_surface(
            "fsaverage", layer="white", join=False
        )
        civet_left, civet_right = fetch_template_surface(
            template, layer="white", join=False
        )

        gradients_fsaverage = np.array(hf["fsaverage"]).T
        gradients_left = _surf2surf(
            fsaverage_left,
            civet_left,
            gradients_fsaverage[: gradients_fsaverage.shape[0] // 2, :],
            interpolation="nearest",
        )
        gradients_right = _surf2surf(
            fsaverage_right,
            civet_right,
            gradients_fsaverage[gradients_fsaverage.shape[0] // 2 :, :],
            interpolation="nearest",
        )
        return np.concatenate((gradients_left, gradients_right), axis=0)
    else:
        return np.array(hf[template]).T


def fetch_yeo_networks_metadata(n: int) -> Tuple[List[str], np.ndarray]:
    """Fetch Yeo networks metadata.

    Parameters
    ----------
    n : int
        Number of Yeo networks, either 7 or 17.

    Returns
    -------
    list of str
        Names of Yeo networks.
    np.ndarray
        Colormap for the Yeo networks.
    """

    if n == 7:
        network_names = [
            "Visual",
            "Somatomotor",
            "Dorsal Attention",
            "Ventral Attention",
            "Limbic",
            "Frontoparietal",
            "Default mode",
        ]
        colormap = (
            np.array(
                [
                    [120, 18, 134],
                    [70, 130, 180],
                    [0, 118, 14],
                    [196, 58, 250],
                    [220, 248, 164],
                    [230, 148, 34],
                    [205, 62, 78],
                ]
            )
            / 255
        )
    elif n == 17:
        network_names = [
            "Visual A",
            "Visual B",
            "Somatomotor A",
            "Somatomotor B",
            "Dorsal Attention A",
            "Dorsal Attention B",
            "Salience / Ventral Attention A",
            "Salience / Ventral Attention B",
            "Limbic A",
            "Limbic B",
            "Frontoparietal C",
            "Frontoparietal A",
            "Frontoparietal B",
            "Temporal Parietal",
            "Default C",
            "Default A",
            "Default B",
        ]
        colormap = (
            np.array(
                [
                    [120, 18, 134],
                    [255, 0, 0],
                    [70, 130, 180],
                    [42, 204, 164],
                    [74, 155, 60],
                    [0, 118, 14],
                    [196, 58, 250],
                    [255, 152, 213],
                    [220, 248, 164],
                    [122, 135, 50],
                    [119, 140, 176],
                    [230, 148, 34],
                    [135, 50, 74],
                    [12, 48, 255],
                    [0, 0, 130],
                    [255, 255, 0],
                    [205, 62, 78],
                ]
            )
            / 255
        )
    else:
        raise ValueError("Invalid number of Yeo networks.")
    return network_names, colormap


def _fetch_template_surface_files(
    template: str,
    data_dir: Union[str, Path],
    layer: Optional[str] = None,
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
        "smoothwm", "pial", "inflated", "sphere" for fsaverage surfaces;
        "midthickness", "inflated", "vinflated" for "fslr32k"; "mid, "white" for
        civet surfaces; and "sphere" for "civet41k" If None, defaults to "pial",
        "midthickness", or "mid", by default None.
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
        if layer == "sphere":
            return _fetch_civet_spheres(template, data_dir=Path(data_dir))
        else:
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
        "cammoun": {
            "n_regions": (33, 60, 125, 250, 500),
            "surfaces": ("fsaverage5", "fsaverage6", "fsaverage", "fslr32k"),
        },
        "glasser": {
            "n_regions": (360,),
            "surfaces": ("fsaverage5", "fsaverage", "fslr32k"),
        },
        "schaefer": {
            "n_regions": (100, 200, 300, 400, 500, 600, 800, 1000),
            "surfaces": ("fsaverage5", "fsaverage6", "fsaverage", "fslr32k"),
        },
        "yeo": {
            "n_regions": (7, 17),
            "surfaces": ("fsaverage5", "fsaverage6", "fsaverage", "fslr32k"),
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
        parcellations[1][parcellations[1] != 0] += n_regions // 2
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


def _fetch_yeo_parcellation(
    template: str, n_regions: int, data_dir: Path
) -> List[np.ndarray]:
    """Fetches Yeo parcellation."""
    filenames = [
        data_dir / f"{template}_{hemi}_yeo{n_regions}.label.gii"
        for hemi in ("lh", "rh")
    ]
    if not all([x.exists() for x in filenames]):
        url = read_data_fetcher_json()["parcellations"]["yeo"]["url"]
        with tempfile.NamedTemporaryFile(suffix=".zip") as f:
            downloaded_file = Path(f.name)
        try:
            _download_file(url, downloaded_file)
            with zipfile.ZipFile(downloaded_file, "r") as zip_ref:
                zip_ref.extractall(data_dir)
        finally:
            downloaded_file.unlink()

    return [nib_load(file).darrays[0].data for file in filenames]


def _fetch_civet_spheres(template: str, data_dir: Path) -> Tuple[str, str]:
    """Fetches CIVET spheres

    Parameters
    ----------
    template : str
        Template name.
    data_dir : Path
        Directory to save the data

    Returns
    -------
    tuple
        Paths to sphere files.
    """

    civet_v2_dir = data_dir / "tpl-civet" / "v2" / template
    civet_v2_dir.mkdir(parents=True, exist_ok=True)

    # Uses the same sphere for L/R hemisphere.
    filename = civet_v2_dir / "tpl-civet_space-ICBM152_sphere.obj"
    if not filename.exists():
        url = read_data_fetcher_json()["spheres"][template]["url"]
        _download_file(url, filename)

    # Return two filenames to conform to other left/right hemisphere functions.
    return (str(filename), str(filename))
