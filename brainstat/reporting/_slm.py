"""HTML reports for fitted SLM objects."""

import html
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

MapSpec = Optional[Sequence[str]]
SurfaceSpec = Optional[Union[str, Mapping[str, object], Sequence[object]]]


def _is_surface_like(surf):
    return all(hasattr(surf, attr) for attr in ("n_points", "Points", "GetCells2D"))


def _fetch_template_surface(slm, name):
    from brainstat.datasets import fetch_template_surface

    return fetch_template_surface(name, join=False, data_dir=slm.data_dir)


def _split_combined_surface(surf):
    from brainspace.mesh.mesh_creation import build_polydata

    if surf.n_points % 2:
        raise ValueError(
            "Cannot infer hemispheres from a combined surface with an odd "
            "number of points. Pass surface=(surf_lh, surf_rh)."
        )
    n_lh = surf.n_points // 2
    points = np.asarray(surf.Points)
    cells = np.asarray(surf.GetCells2D())
    lh_cells = cells[np.all(cells < n_lh, axis=1)]
    rh_cells = cells[np.all(cells >= n_lh, axis=1)] - n_lh
    if (
        lh_cells.size == 0
        or rh_cells.size == 0
        or len(lh_cells) + len(rh_cells) != len(cells)
    ):
        raise ValueError(
            "Cannot infer separate left/right cells from the combined surface. "
            "Pass surface=(surf_lh, surf_rh)."
        )
    return (
        build_polydata(points[:n_lh], lh_cells),
        build_polydata(points[n_lh:], rh_cells),
    )


def _resolve_surfaces(slm, surface: SurfaceSpec):
    if isinstance(surface, str):
        return _fetch_template_surface(slm, surface)
    if isinstance(surface, Mapping):
        return {"lh": surface["lh"], "rh": surface["rh"]}
    if isinstance(surface, (list, tuple)) and len(surface) == 2:
        return {"lh": surface[0], "rh": surface[1]}
    if _is_surface_like(surface):
        lh, rh = _split_combined_surface(surface)
        return {"lh": lh, "rh": rh}
    if surface is not None:
        raise TypeError(
            "surface must be a template name, BSPolyData pair, or {'lh', 'rh'} dict."
        )

    if getattr(slm, "surf_name", None):
        return _fetch_template_surface(slm, slm.surf_name)
    if _is_surface_like(getattr(slm, "surf", None)):
        lh, rh = _split_combined_surface(slm.surf)
        return {"lh": lh, "rh": rh}
    if isinstance(getattr(slm, "surf", None), Mapping):
        raise ValueError(
            "SLM.surf is a combined SurfStat-style dictionary. Pass "
            "surface=(surf_lh, surf_rh) or surface='template_name' to write "
            "an HTML report."
        )
    raise ValueError(
        "No surface is available for this SLM. Pass surface='fsaverage5' or "
        "surface=(surf_lh, surf_rh)."
    )


def _as_map(values, name):
    arr = np.asarray(values, dtype=float).squeeze()
    if arr.ndim == 0:
        raise ValueError("SLM map {!r} is scalar and cannot be plotted.".format(name))
    if arr.ndim > 1:
        raise ValueError(
            "SLM map {!r} has shape {}; only univariate maps are supported "
            "in the first HTML report implementation.".format(name, arr.shape)
        )
    return arr.ravel()


def _pval_map(slm, key):
    pval = getattr(slm, "P", None)
    if not isinstance(pval, Mapping):
        return None
    return (pval.get("pval") or {}).get(key)


def _map_candidates(slm):
    candidates = {
        "t": ("t-statistic", getattr(slm, "t", None)),
        "p": ("RFT vertex p-values", _pval_map(slm, "P")),
        "cluster_p": ("RFT cluster p-values", _pval_map(slm, "C")),
        "q": ("FDR q-values", getattr(slm, "Q", None)),
    }
    return candidates


def _slm_maps(slm, maps: MapSpec):
    candidates = _map_candidates(slm)
    requested = (
        ([maps] if isinstance(maps, str) else list(maps))
        if maps is not None
        else ["t", "p", "cluster_p", "q"]
    )
    explicit = maps is not None
    out = {}
    for key in requested:
        if key not in candidates:
            raise ValueError(
                "Unknown SLM report map {!r}. Valid maps are: {}.".format(
                    key, ", ".join(sorted(candidates))
                )
            )
        label, values = candidates[key]
        if values is None:
            if explicit:
                raise ValueError("SLM report map {!r} is not available.".format(key))
            continue
        out[label] = _as_map(values, key)
    if not out:
        raise ValueError("No fitted SLM maps are available to report.")
    return out


def _metadata(slm, included_maps):
    import brainspace
    import brainstat

    correction = getattr(slm, "correction", None)
    if isinstance(correction, str):
        correction = [correction]
    return {
        "BrainStat version": brainstat.__version__,
        "BrainSpace version": brainspace.__version__,
        "Surface": getattr(slm, "surf_name", None) or "provided surface",
        "Corrections": ", ".join(correction) if correction else "none",
        "Two-tailed": getattr(slm, "two_tailed", None),
        "Cluster threshold": getattr(slm, "cluster_threshold", None),
        "Maps": ", ".join(included_maps),
        "Contrast": np.asarray(getattr(slm, "contrast", [])).tolist(),
        "Design shape": list(
            getattr(getattr(slm, "model", None), "matrix", np.empty((0, 0))).shape
        ),
    }


def make_slm_report(
    slm,
    filename="slm_report.html",
    maps: MapSpec = None,
    surface: SurfaceSpec = None,
    title="BrainStat SLM Report",
    cmap="viridis",
    color_range=None,
) -> Path:
    """Write a display-independent HTML report for a fitted SLM.

    Parameters
    ----------
    slm : brainstat.stats.SLM.SLM
        Fitted SLM object.
    filename : str or path-like, optional
        Output HTML path.
    maps : sequence of {'t', 'p', 'cluster_p', 'q'}, optional
        Maps to include. If omitted, all available maps are included.
    surface : str, pair, dict or BSPolyData, optional
        Surface used for the report. Template names are fetched as split
        hemispheres. Combined surfaces are split when possible.
    title : str, optional
        Report title.
    cmap : str, optional
        Matplotlib colormap for the embedded Nilearn surface views.
    color_range : {'sym'}, tuple or None, optional
        Color range for each map. By default, p/q maps use (0, 1).

    Returns
    -------
    pathlib.Path
        Path to the written report.
    """
    from matplotlib.pyplot import get_cmap
    from nilearn.plotting import view_surf

    if getattr(slm, "t", None) is None:
        raise ValueError("Fit the SLM before creating a report.")
    surfaces = _resolve_surfaces(slm, surface)
    if not isinstance(surfaces, Mapping):
        surfaces = dict(zip(("lh", "rh"), surfaces))
    hemispheres = [surfaces[key] for key in ("lh", "rh")]
    points = [np.asarray(surf.Points) for surf in hemispheres]
    faces = [np.asarray(surf.GetCells2D()) for surf in hemispheres]
    n_vertices = sum(len(part) for part in points)
    if any(face.ndim != 2 or face.shape[1] != 3 for face in faces):
        raise ValueError("HTML reports require triangular surface geometry.")
    mesh = (np.vstack(points), np.vstack((faces[0], faces[1] + len(points[0]))))
    report_maps = _slm_maps(slm, maps)
    mask = getattr(slm, "mask", None)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or mask.size != n_vertices:
            raise ValueError("The SLM mask must match the report surface vertices.")
    metadata = _metadata(slm, report_maps.keys())
    metadata["Vertices"] = n_vertices
    metadata["Included vertices"] = int(mask.sum()) if mask is not None else n_vertices
    metadata["Surface"] = surface if isinstance(surface, str) else metadata["Surface"]
    panels, payload_maps = [], []
    palette = get_cmap(cmap)
    for index, (label, values) in enumerate(report_maps.items()):
        if values.size != n_vertices:
            raise ValueError(
                f"Map {label!r} has {values.size} values but the surface has "
                f"{n_vertices} vertices. Expand parcellated maps to vertices first."
            )
        values = values.copy()
        if mask is not None:
            values[~mask] = np.nan
        if not np.any(np.isfinite(values)):
            raise ValueError(f"Map {label!r} has no finite values to display.")
        if np.any(np.isinf(values)):
            raise ValueError(f"Map {label!r} contains infinite values.")
        finite = values[np.isfinite(values)]
        symmetric = (color_range is None and label == "t-statistic") or (
            isinstance(color_range, str) and color_range == "sym"
        )
        limits = dict(vmax=max(float(np.abs(finite).max()), np.finfo(float).eps))
        if color_range is not None and not symmetric:
            if isinstance(color_range, str) or len(color_range) != 2:
                raise ValueError(
                    "color_range must be None, 'sym', or a (min, max) pair."
                )
            limits = dict(vmin=color_range[0], vmax=color_range[1])
        elif color_range is None and label != "t-statistic":
            limits = dict(vmin=0, vmax=1)
        view = view_surf(
            mesh,
            surf_map=values,
            cmap=palette,
            symmetric_cmap=symmetric,
            title=label,
            **limits,
        )
        panels.append(
            f'<iframe title="{html.escape(label, quote=True)}" '
            f'srcdoc="{html.escape(view.html, quote=True)}" '
            f'loading="lazy" class="map" id="map-{index}" '
            f'{"" if index == 0 else "hidden"}></iframe>'
        )
        payload_maps.append(
            dict(
                name=label,
                values=[
                    float(value) if np.isfinite(value) else None for value in values
                ],
            )
        )
    payload = json.dumps(
        dict(metadata=metadata, maps=payload_maps), allow_nan=False
    ).replace("<", "\\u003c")
    options = "".join(
        f'<option value="map-{index}">{html.escape(label)}</option>'
        for index, label in enumerate(report_maps)
    )
    details = "".join(
        f"<dt>{html.escape(key)}</dt><dd>{html.escape(str(value))}</dd>"
        for key, value in metadata.items()
    )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:16px system-ui;margin:2rem auto;max-width:1200px;padding:0 1rem;"
        "color:#18242c;background:#f8fafb}iframe{width:100%;height:620px;border:0;background:white}"
        "[hidden]{display:none}select{font:inherit;padding:.5rem}dl{display:grid;"
        "grid-template-columns:12rem 1fr;gap:.5rem}dd{margin:0;overflow-wrap:anywhere}</style>"
        f"<h1>{html.escape(title)}</h1><p>Interactive fitted surface maps. "
        "P and q maps are unthresholded; NaN values mark excluded or missing vertices.</p>"
        f'<label for="map-select">Map </label><select id="map-select">{options}</select>'
        + "".join(panels)
        + f"<details><summary>Model and provenance</summary><dl>{details}</dl></details>"
        + '<script id="brainstat-data" type="application/json">'
        + payload
        + "</script>"
        + '<script>document.getElementById("map-select").addEventListener("change",function(){'
        'document.querySelectorAll("iframe.map").forEach(p=>{p.hidden=p.id!==this.value;});'
        'window.dispatchEvent(new Event("resize"));});</script></html>'
    )
    output = Path(filename)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, suffix=".html", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(document)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output
