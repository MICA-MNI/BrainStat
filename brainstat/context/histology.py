""" Histology context decoder """
from pathlib import Path
import logging
import urllib.request
import shutil
import pandas as pd
import numpy as np
import h5py
import hcp_utils as hcp

from sklearn.linear_model import LinearRegression
from nilearn import datasets
from brainspace.gradient.gradient import GradientMaps
from brainspace.mesh.mesh_creation import build_polydata
from brainspace.mesh.mesh_elements import get_points
from brainspace.utils.parcellation import reduce_by_labels
from pingouin import pcorr
from .utils import read_surface_gz


def compute_histology_gradients(
    mpc,
    kernel="cosine",
    approach="dm",
    n_components=10,
    alignment=None,
    random_state=None,
    gamma=None,
    sparsity=0.9,
    reference=None,
    n_iter=10,
):
    """Computes microstructural profile covariance gradients.

    Parameters
    ----------
    mpc : numpy.ndarray
        Microstructural profile covariance matrix.
    kernel : str, optional
         Kernel function to build the affinity matrix. Possible options: {‘pearson’,
         ‘spearman’, ‘cosine’, ‘normalized_angle’, ‘gaussian’}. If callable, must
         receive a 2D array and return a 2D square array. If None, use input matrix.
         By default "cosine".
    approach : str, optional
        Embedding approach. Can be 'pca' for Principal Component Analysis, 'le' for
        laplacian eigenmaps, or 'dm' for diffusion mapping, by default "dm".
    n_components : int, optional
        Number of components to return, by default 10.
    alignment : str, None, optional
        Alignment approach. Only used when two or more datasets are provided.
        Valid options are 'pa' for procrustes analysis and "joint" for joint embedding.
        If None, no alignment is peformed, by default None.
    random_state : int, None, optional
        Random state, by default None
    gamma : float, None, optional
        Inverse kernel width. Only used if kernel == "gaussian". If None, gamma=1/n_feat,
        by default None.
    sparsity : float, optional
        Proportion of smallest elements to zero-out for each row, by default 0.9.
    reference : numpy.ndarray, optional
        Initial reference for procrustes alignments. Only used when
        alignment == 'procrustes', by default None.
    n_iter : int, optional
        Number of iterations for Procrustes alignment, by default 10.

    Returns
    -------
    brainspace.gradient.gradient.GradientMaps
        BrainSpace gradient maps object.
    """
    gm = GradientMaps(
        kernel=kernel,
        approach=approach,
        n_components=n_components,
        alignment=alignment,
        random_state=random_state,
    )
    gm.fit(mpc, gamma=gamma, sparsity=sparsity, n_iter=n_iter, reference=reference)
    return gm


def compute_mpc(profile, labels, template=None, regress_y=False):
    """Computes MPC for given labels on a surface template.

    Parameters
    ----------
    profile : numpy.ndarray
        Histological profiles.
    labels : numpy.ndarray
        Labels of regions of interest.
    template : str, None, optional
        Surface template, either 'fsaverage' or 'fs_LR' or a list of two strings
        containing paths to the left/right (in that order) hemispheres, by default
        'fsaverage'. This argument is required if regress_y==True.
    regress_y : bool, optional
        If true, regress the y-coordinate from the profiels, by default True.

    Returns
    -------
    numpy.ndarray
        Microstructural profile covariance.
    """

    if regress_y:
        if template is None:
            raise ValueError("If regress_y is True, template may not be None.")
        profile = y_correction(profile, template)

    roi_profile = reduce_by_labels(profile, labels)
    partial_correlation = pcorr(pd.DataFrame(roi_profile)).to_numpy()

    mpc = 0.5 * np.log((1 + partial_correlation) / (1 - partial_correlation))
    mpc[mpc == np.inf] = 0
    mpc[mpc == np.nan] = 0
    return mpc


def y_correction(profile, template):
    """Regresses y-coordinate from profiles and returns residuals.

    Parameters
    ----------
    profile : numpy.ndarray
        BigBrain intensity profiles.
    template : str, list
        Template name ('fs_LR', 'fsaverage') or a list containing two filenames
        of left and right hemispheric surfaces.

    Returns
    -------
    numpy.ndarray
        Residuals.

    Raises
    ------
    ValueError
        Throws an error if template is None.
    """

    if isinstance(template, str):
        surfaces = template_to_surfaces(template)
    else:
        surfaces = [read_surface_gz(x) for x in template]

    coordinates = [np.array(get_points(x)) for x in surfaces]
    coordinates = np.concatenate(coordinates)

    model = LinearRegression().fit(coordinates[:, 1][:, None], profile)
    residuals = profile - model.predict(coordinates[:, 1])
    return residuals


def template_to_surfaces(template):
    """Converts a template string to BrainSpace surfaces.

    Parameters
    ----------
    template : str
        'fsaverage' or 'fs_LR'.

    Returns
    -------
    list
        BrainSpace surfaces. First element is the left hemisphere.
    """

    if isinstance(template, str):
        # Assume template name.
        if template == "fsaverage":
            fsaverage = datasets.fetch_surf_fsaverage()
            surfaces = [
                read_surface_gz(fsaverage["pial_left"]),
                read_surface_gz(fsaverage["pial_right"]),
            ]
        elif template == "fs_LR":
            surfaces_hcp = [hcp.mesh["pial_left"], hcp.mesh["pial_right"]]
            surfaces = [build_polydata(x[0], x[1]) for x in surfaces_hcp]

    return surfaces


def read_histology_profile(data_dir=None, template="fsaverage", overwrite=False):
    """Reads BigBrain histology profiles.

    Parameters
    ----------
    data_dir : str, None, optional
        Path to the data directory. If data is not found here then data will be
        downloaded. If None, data_dir is set to the home directory, by default None.
    template : str, optional
        Surface template. Currently allowed options are 'fsaverage' and 'fs_LR', by
        default 'fsaverage'.
    overwrite : bool, optional
        If true, existing data will be overwrriten, by default False.

    Returns
    -------
    numpy.ndarray
        Depth-by-vertex array of BigBrain intensities.
    """

    if data_dir is None:
        data_dir = Path.home() / "histology_data"
    else:
        data_dir = Path(data_dir)
    histology_file = data_dir / ("histology_" + template + ".h5")

    if not histology_file.exists() or overwrite:
        logging.info(
            "Could not find a histological profile or an overwrite was requested. Downloading..."
        )
        download_histology_profiles(
            data_dir=data_dir, template=template, overwrite=overwrite
        )

    import pdb

    pdb.set_trace()
    with h5py.File(histology_file, "r") as h5_file:
        return h5_file.get(template)[...]


def download_histology_profiles(data_dir=None, template="fsaverage", overwrite=False):
    """Downloads BigBrain histology profiles.

    Parameters
    ----------
    data_dir : str, None, optional
        Path to the directory to store the data. If None, defaults to the home
        directory, by default None.
    template : str, optional
        Surface template. Currently allowed options are 'fsaverage' and 'fs_LR', by
        default 'fsaverage'.
    overwrite : bool, optional
        If true, existing data will be overwrriten, by default False.

    Raises
    ------
    KeyError
        Thrown if an invalid template is requested.
    """

    if data_dir is None:
        data_dir = Path.home() / "histology_data"
    else:
        data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / ("histology_" + template + ".h5")

    urls = {
        "fsaverage": "https://box.bic.mni.mcgill.ca/s/znBp7Emls0mMW1a/download",
        "fs_LR": "https://box.bic.mni.mcgill.ca/s/ge1GVQJ0YFwtYLL/download",
    }

    try:
        _download_file(urls[template], output_file, overwrite)
    except KeyError:
        raise KeyError(
            "Could not find the requested template. Valid templates are: 'fs_LR', 'fsaverage'."
        )


def _download_file(url, output_file, overwrite):
    """Downloads a file.

    Parameters
    ----------
    url : str
        URL of the download.
    file : pathlib.Path
        Path object of the output file.
    overwrite : bool
        If true, overwrite existing files.
    """

    if output_file.exists() and not overwrite:
        logging.debug(str(output_file) + " already exists and will not be overwritten.")
        return

    logging.debug("Downloading " + str(output_file))
    with urllib.request.urlopen(url) as response, open(output_file, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def __generate_histology_files(BigBrainWarpPath, output_dir):
    """Generates the histology files from BigBrainWarp

    Parameters
    ----------
    BigBrainWarpPath : str, pathlib.Path
        Path to the BigBrainWarp repository.
    output_dir : str, pathlib.Path
        Path to the output directory.

    Notes
    -----
    Function is for internal usage only. It is included here merely for
    users who wish to inspect how the files were created. It is recommended
    to use other functions in this module for downloading the output files.
    Other templates were created from these using Workbench Tools.
    """

    base_dir = Path(BigBrainWarpPath) / "spaces"
    for template in ["fsaverage", "fs_LR"]:
        profile_file = base_dir / template / ("profiles_" + template + ".txt")
        output_file = Path(output_dir) / ("histology_" + template + ".h5")

        profile = pd.read_csv(profile_file, header=None)
        with h5py.File(output_file, "w") as h5_file:
            h5_file.create_dataset(
                template,
                data=profile.to_numpy(),
                compression="gzip",
                compression_opts=9,
            )


def __warp_histology_files(
    source_spheres,
    target_spheres,
    output_file,
    template_name='fsaverage',
    method="BARYCENTRIC",
    source_area=None,
    target_area=None,
):
    """Warps histology profiles to a different template.

    Parameters
    ----------
    source_spheres : list
        Path to the source sphere. Left must be provided first.
    target_spheres : list
        Path to the target sphere. Left must be provided first.
    output_file : str
        Path to the output file.
    template_name : str
        Name of the input template, by default 'fsaverage'.
    method : str, optional
        Interpolation method. Must be either 'ADAP_BARY_AREA' or 'BARYCENTRIC',
        by default 'BARYCENTRIC'.
    current_area : str, optional
        Path to a metric file with vertex areas for the source sphere. Only used
        and obligatory if method=='ADAP_BARY_AREA'.
    target_area : str, optional
        Path to a metric file with vertex areas for the target sphere. Only used
        and obligatory if method=='ADAP_BARY_AREA'.

    Notes
    -----
    This function is for internal usage only. It is included here merely for
    users who wish to inspect how the files were created. It is recommended
    to use other functions in this module for downloading the output files.
    """

    import subprocess
    import tempfile
    from nibabel.gifti.gifti import GiftiImage, GiftiDataArray
    from nibabel import save, load

    profiles = read_histology_profile('fsaverage')

    data = [GiftiImage(), GiftiImage()]
    data[0].add_gifti_data_array(
        GiftiDataArray(data=profiles[:, : profiles.shape[1] // 2])
    )
    data[1].add_gifti_data_array(
        GiftiDataArray(data=profiles[:, profiles.shape[1] // 2 :])
    )

    for i in range(len(data)):
        with tempfile.NamedTemporaryFile(suffix=".func.gii") as metric_file:
            with tempfile.NamedTemporaryFile(suffix=".func.gii") as target_file:
                save(data[i], metric_file.name)
                if method == "BARYCENTRIC":
                    subprocess.run(
                        [
                            "wb_command",
                            "-metric-resample",
                            metric_file.name,
                            source_spheres[i],
                            target_spheres[i],
                            method,
                            target_file,
                        ]
                    )
                else:
                    subprocess.run(
                        [
                            "wb_command",
                            "-metric-resample",
                            metric_file.name,
                            source_spheres[i],
                            target_spheres[i],
                            method,
                            target_file,
                            "-area-metrics",
                            source_area,
                            target_area,
                        ]
                    )
                gifti = load(target_file)
                import pdb
                pdb.set_trace()
