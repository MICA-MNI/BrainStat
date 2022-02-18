"""Operations on meshes."""

import sys
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
from brainspace.mesh.mesh_elements import get_edges
from brainspace.vtk_interface.wrappers.data_object import BSPolyData
from nibabel.nifti1 import Nifti1Image

from brainstat._typing import ArrayLike, NDArray
from brainstat.stats.utils import colon

if TYPE_CHECKING:
    from brainstat.stats.SLM import SLM  # type: ignore


def mesh_edges(
    surf: Union[dict, BSPolyData, "SLM", Nifti1Image], mask: Optional[ArrayLike] = None
) -> np.ndarray:
    """Converts the triangles or lattices of a mesh to edges.

    Parameters
    ----------
        surf : dict, BSPolyData, SLM, Nifti1Image
            One of the following:
             - A dictionary with key 'tri' where tri is numpy array of triangle indices, t:#triangles.
                Note that, for compatibility with SurfStat, these triangles are 1-indexed, not 0-indexed.
             - A dictionary with key 'lat' where lat is a 3D numpy array of 1's and 0's (1:in, 0:out).
             - A BrainSpace surface object
             - An SLM object with an associated surface.

    Returns
    -------
        np.ndarray
            A e-by-2 numpy array containing the indices of the edges, where
            e is the number of edges. Note that these are 0-indexed.
    """

    # Convert all inputs to a compatible dictionary, retain BSPolyData.
    if type(surf).__name__ == "SLM":
        if surf.tri is not None:  # type: ignore
            edg = triangles_to_edges(surf.tri)  # type: ignore
        elif surf.lat is not None:  # type: ignore
            edg = lattice_to_edges(surf.lat)  # type: ignore
        else:
            ValueError("SLM object does not have triangle/lattice data.")
    elif isinstance(surf, dict):
        if "tri" in surf:
            edg = triangles_to_edges(surf["tri"])
        if "lat" in surf:
            edg = lattice_to_edges(surf["lat"])
    elif isinstance(surf, Nifti1Image):
        if mask is not None:
            raise ValueError(
                "Masks are currently not compatible with a NIFTI image lattice input."
            )
        edg = lattice_to_edges(surf.get_fdata() != 0)
    elif isinstance(surf, BSPolyData):
        edg = get_edges(surf)
    else:
        raise ValueError("Unknown surface format.")

    if mask is not None:
        edg, _ = _mask_edges(edg, mask)

    return edg


def _mask_edges(edges: np.ndarray, mask: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Masks edges based on a mask.

    Parameters
    ----------
    edges : np.ndarray
        A e-by-2 numpy array containing the indices of the edges, where
        e is the number of edges.
    mask : np.ndarray
        A m-by-n numpy array of 1's and 0's (1:in, 0:out).

    Returns
    -------
    np.ndarray
        A e-by-2 numpy array containing the indices of the masked edges, where e
        is the number of edges.
    np.ndarray
        Array of indices of the retained edges.
    """

    missing_edges = np.where(np.logical_not(mask))
    remove_edges = np.isin(edges, missing_edges)
    idx = ~np.any(remove_edges, axis=1)
    edges_new = edges[idx, :]
    edges_new = _make_contiguous(edges_new)
    return edges_new, idx


def _make_contiguous(Y: np.ndarray) -> np.ndarray:
    """Makes values of Y contiguous integers

    Parameters
    ----------
    Y : numpy.ndarray
        Array with uncontiguous numbers.

    Returns
    -------
    numpy.ndarray
        Array Y converted to contiguous numbers in range(np.unique(Y).size).
    """
    Y_flat = Y.copy().flatten()
    val = np.unique(Y_flat)
    new_val = np.arange(val.size)

    D = dict(np.array([val, new_val]).T)
    Y_new = [D[i] for i in Y_flat]
    return np.reshape(Y_new, Y.shape)


def triangles_to_edges(tri: ArrayLike) -> np.ndarray:
    """Convert a triangular mesh to an edge list.

    Parameters
    ----------
    tri : numpy.ndarray
        Array of shape (n, 3) with the indices of the vertices of the triangles.

    Returns
    -------
    numpy.ndarray
        Array of shape (m, 2) with the indices of the edges.

    """
    tri = np.sort(tri, axis=1)
    edg = np.unique(
        np.concatenate(
            (np.concatenate((tri[:, [0, 1]], tri[:, [0, 2]])), tri[:, [1, 2]])
        ),
        axis=0,
    )
    return edg - 1


def lattice_to_edges(lattice: NDArray) -> np.ndarray:
    # See the comments of SurfStatResels for a full explanation.
    if lattice.ndim == 2:
        lattice = np.expand_dims(lattice, axis=2)

    I, J, K = np.shape(lattice)
    IJ = I * J

    a = np.arange(1, int(I) + 1, dtype="int")
    b = np.arange(1, int(J) + 1, dtype="int")

    i, j = np.meshgrid(a, b)
    i = i.T.flatten("F")
    j = j.T.flatten("F")

    n1 = (I - 1) * (J - 1) * 6 + (I - 1) * 3 + (J - 1) * 3 + 1
    n2 = (I - 1) * (J - 1) * 3 + (I - 1) + (J - 1)

    edg = np.zeros(((K - 1) * n1 + n2, int(2)), dtype="int")

    for f in range(0, 2):

        c1 = np.where((np.remainder((i + j), 2) == f) & (i < I) & (j < J))[0]
        c2 = np.where((np.remainder((i + j), 2) == f) & (i > 1) & (j < J))[0]
        c11 = np.where((np.remainder((i + j), 2) == f) & (i == I) & (j < J))[0]
        c21 = np.where((np.remainder((i + j), 2) == f) & (i == I) & (j > 1))[0]
        c12 = np.where((np.remainder((i + j), 2) == f) & (i < I) & (j == J))[0]
        c22 = np.where((np.remainder((i + j), 2) == f) & (i > 1) & (j == J))[0]

        # bottom slice
        edg0 = (
            np.block(
                [
                    [c1, c1, c1, c2 - 1, c2 - 1, c2, c11, c21 - I, c12, c22 - 1],
                    [
                        c1 + 1,
                        c1 + I,
                        c1 + 1 + I,
                        c2,
                        c2 - 1 + I,
                        c2 - 1 + I,
                        c11 + I,
                        c21,
                        c12 + 1,
                        c22,
                    ],
                ]
            ).T
            + 1
        )
        # between slices
        edg1 = (
            np.block(
                [
                    [c1, c1, c1, c11, c11, c12, c12],
                    [
                        c1 + IJ,
                        c1 + 1 + IJ,
                        c1 + I + IJ,
                        c11 + IJ,
                        c11 + I + IJ,
                        c12 + IJ,
                        c12 + 1 + IJ,
                    ],
                ]
            ).T
            + 1
        )

        edg2 = (
            np.block(
                [
                    [c2 - 1, c2, c2 - 1 + I, c21 - I, c21, c22 - 1, c22],
                    [
                        c2 - 1 + IJ,
                        c2 - 1 + IJ,
                        c2 - 1 + IJ,
                        c21 - I + IJ,
                        c21 - I + IJ,
                        c22 - 1 + IJ,
                        c22 - 1 + IJ,
                    ],
                ]
            ).T
            + 1
        )

        if f:
            for k in colon(2, K - 1, 2):
                edg[(k - 1) * n1 + np.arange(0, n1), :] = (
                    np.block([[edg0], [edg2], [edg1], [IJ, 2 * IJ]]) + (k - 1) * IJ  # type: ignore
                )

        else:
            for k in colon(1, K - 1, 2):
                edg[(k - 1) * n1 + np.arange(0, n1), :] = (
                    np.block([[edg0], [edg1], [edg2], [IJ, 2 * IJ]]) + (k - 1) * IJ  # type: ignore
                )

        if np.remainder((K + 1), 2) == f:
            # top slice
            edg[(K - 1) * n1 + np.arange(0, n2), :] = (
                edg0[np.arange(0, n2), :] + (K - 1) * IJ
            )

    # index by voxels in the "lat"
    vid = np.array(
        np.multiply(np.cumsum(lattice[:].T.flatten()), lattice[:].T.flatten()),
        dtype="int",
    )
    vid = vid.reshape(len(vid), 1)

    # only inside the lat
    all_idx = np.all(
        np.block(
            [
                [lattice.T.flatten()[edg[:, 0] - 1]],
                [lattice.T.flatten()[edg[:, 1] - 1]],
            ]
        ).T,
        axis=1,
    )

    edg = vid[edg[all_idx, :] - 1].reshape(np.shape(edg[all_idx, :] - 1))
    return edg - 1
