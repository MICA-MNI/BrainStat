"""Operations on meshes."""

import sys
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
from brainspace.mesh.mesh_elements import get_edges
from brainspace.vtk_interface.wrappers.data_object import BSPolyData

from brainstat._typing import ArrayLike
from brainstat.stats.utils import colon

if TYPE_CHECKING:
    from brainstat.stats.SLM import SLM  # type: ignore


def mesh_edges(
    surf: Union[dict, BSPolyData, "SLM"], mask: Optional[ArrayLike] = None
) -> np.ndarray:
    """Converts the triangles or lattices of a mesh to edges.

    Args:
        surf (dict): = a dictionary with key 'tri' or 'lat'
        surf['tri'] = (t x 3) numpy array of triangle indices, t:#triangles, or,
        surf['lat'] = 3D numpy array of 1's and 0's (1:in, 0:out).
        or
        surf (BSPolyData) = a BrainSpace surface object
        or
        surf (SLM) = a SLM object with an associated surface.

    Returns:
        edg (np.array): A e-by-2 numpy array containing the indices of the edges, where
        e is the number of edges.
    """

    # This doesn't strictly test that its BrainStat SLM, but we can't import
    # directly without causing a circular import.
    if not isinstance(surf, dict) and not isinstance(surf, BSPolyData):
        if surf.tri is not None:
            surf = {"tri": surf.tri}
        elif surf.lat is not None:
            surf = {"lat": surf.lat}
        elif surf.surf is not None:
            return mesh_edges(surf.surf)
        else:
            ValueError("SLM object does not have triangle/lattice data.")

    # For BSPolyData, simply use BrainSpace's functionality to grab edges.
    if isinstance(surf, BSPolyData):
        edg = get_edges(surf)

    # Convert triangles to edges by grabbing all unique edges within triangles.
    elif "tri" in surf:
        tri = np.sort(surf["tri"], axis=1)
        edg = np.unique(
            np.concatenate(
                (np.concatenate((tri[:, [0, 1]], tri[:, [0, 2]])), tri[:, [1, 2]])
            ),
            axis=0,
        )
        edg = edg - 1

    elif "lat" in surf:
        # See the comments of SurfStatResels for a full explanation.
        if surf["lat"].ndim == 2:
            surf["lat"] = np.expand_dims(surf["lat"], axis=2)

        I, J, K = np.shape(surf["lat"])
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
            np.multiply(
                np.cumsum(surf["lat"][:].T.flatten()), surf["lat"][:].T.flatten()
            ),
            dtype="int",
        )
        vid = vid.reshape(len(vid), 1)

        # only inside the lat
        all_idx = np.all(
            np.block(
                [
                    [surf["lat"].T.flatten()[edg[:, 0] - 1]],
                    [surf["lat"].T.flatten()[edg[:, 1] - 1]],
                ]
            ).T,
            axis=1,
        )

        edg = vid[edg[all_idx, :] - 1].reshape(np.shape(edg[all_idx, :] - 1))
        edg = edg - 1

    else:
        sys.exit('Input "surf" must have "lat" or "tri" key, or be a mesh object.')

    if mask is not None:
        edg, _ = _mask_edges(edg, mask)

    return edg


def _mask_edges(edges: np.ndarray, mask: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    # TODO: this section is sloppily written.
    missing_edges = np.where(np.logical_not(mask))
    remove_edges = np.zeros(edges.shape, dtype=bool)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            remove_edges[i, j] = (edges[i, j] == missing_edges).any()
    idx = ~np.any(remove_edges, axis=1)
    edges = edges[idx, :]
    edges = _make_contiguous(edges)
    return edges, idx


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
    val = np.unique(Y)
    for i in range(val.size):
        Y[Y == val[i]] = i
    return Y
