"""Real display-free report generation and scientific map contracts."""

import html
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
from brainspace.mesh.mesh_creation import build_polydata
from brainspace.mesh.mesh_operations import combine_surfaces

from brainstat.reporting import make_slm_report
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect


def _surfaces():
    cells = np.array([[0, 1, 2]])
    lh = build_polydata(np.array([[-1.0, 0, 0], [-0.2, 0, 0], [-0.6, 0.7, 0]]), cells)
    rh = build_polydata(np.array([[0.2, 0, 0], [1.0, 0, 0], [0.6, 0.7, 0]]), cells)
    return lh, rh


def _slm():
    slm = SLM(FixedEffect(np.ones(4)), np.ones(4), correction=["fdr"])
    slm.t = np.arange(6, dtype=float)
    slm.Q = np.linspace(0.01, 0.06, 6)
    slm.P = {"pval": {"P": np.linspace(0.02, 0.07, 6), "C": None}}
    return slm


def _payload(path):
    text = path.read_text()
    match = re.search(
        r'<script id="brainstat-data" type="application/json">(.*?)</script>', text
    )
    return json.loads(match.group(1))


def test_report_serializes_maps_and_masks_without_changing_slm(tmp_path):
    slm = _slm()
    slm.mask = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    original = slm.t.copy()
    path = slm.to_html_report(tmp_path / "report.html", surface=_surfaces())
    payload = _payload(path)
    assert [entry["name"] for entry in payload["maps"]] == [
        "t-statistic",
        "RFT vertex p-values",
        "FDR q-values",
    ]
    assert payload["maps"][0]["values"] == [0.0, 1.0, None, 3.0, 4.0, 5.0]
    assert payload["metadata"]["Included vertices"] == 5
    assert payload["metadata"]["Corrections"] == "fdr"
    np.testing.assert_array_equal(slm.t, original)
    content = path.read_text()
    assert "srcdoc=" in content and "plotly" in content.lower()
    # Nilearn bundles its browser runtime; there must be no script downloads.
    embedded = html.unescape(content)
    assert re.search(r'<script[^>]+src=["\']https?://', embedded) is None


@pytest.mark.parametrize("kind", ["pair", "mapping", "combined", "template", "stored"])
def test_surface_resolution(tmp_path, monkeypatch, kind):
    surfaces = _surfaces()
    slm = _slm()
    monkeypatch.setattr(
        "brainstat.reporting._slm._fetch_template_surface", lambda *a: surfaces
    )
    surface = {
        "pair": surfaces,
        "mapping": dict(zip(("lh", "rh"), surfaces)),
        "combined": combine_surfaces(*surfaces),
        "template": "fsaverage5",
        "stored": None,
    }[kind]
    if kind == "stored":
        slm.surf = combine_surfaces(*surfaces)
    path = make_slm_report(slm, tmp_path / f"{kind}.html", maps="t", surface=surface)
    assert len(_payload(path)["maps"]) == 1


@pytest.mark.parametrize(
    "maps,error",
    [(["q", "bad"], "Unknown"), (["cluster_p"], "not available"), ([], "No fitted")],
)
def test_bad_map_requests_preserve_existing_file(tmp_path, maps, error):
    path = tmp_path / "report.html"
    path.write_text("existing")
    with pytest.raises(ValueError, match=error):
        make_slm_report(_slm(), path, maps=maps, surface=_surfaces())
    assert path.read_text() == "existing"


def test_vertex_mismatch_and_unfitted_model_rejected(tmp_path):
    slm = _slm()
    slm.t = np.ones(3)
    with pytest.raises(ValueError, match="Expand parcellated"):
        make_slm_report(slm, tmp_path / "bad.html", maps="t", surface=_surfaces())
    slm.t = None
    with pytest.raises(ValueError, match="Fit the SLM"):
        make_slm_report(slm, tmp_path / "bad.html", surface=_surfaces())


def test_fitted_slm_export_without_display(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    rng = np.random.default_rng(0)
    age = np.arange(10.0)
    slm = SLM(FixedEffect(age), age, correction="fdr")
    slm.fit(rng.normal(size=(10, 6)))
    path = slm.to_html_report(tmp_path / "fitted.html", surface=_surfaces())
    payload = _payload(path)
    assert [entry["name"] for entry in payload["maps"]] == [
        "t-statistic",
        "FDR q-values",
    ]
    np.testing.assert_allclose(payload["maps"][0]["values"], slm.t.ravel())
    assert not any("brainspace.plotting.base" == key for key in sys.modules)


def test_report_escapes_titles_and_metadata(tmp_path):
    title = '</script><script>alert("x")</script>'
    slm = _slm()
    slm.surf_name = title
    path = make_slm_report(
        slm, tmp_path / "escape.html", surface=_surfaces(), maps="t", title=title
    )
    assert title not in path.read_text()
    assert _payload(path)["metadata"]["Surface"] == title


def test_reporting_import_does_not_load_vtk_renderer():
    code = "from brainstat.reporting import make_slm_report; import sys; assert 'vtk' not in sys.modules; assert 'brainspace.plotting.base' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
