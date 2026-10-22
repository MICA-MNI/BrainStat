"""Offline contracts for genetics; Allen downloads are an opt-in integration test."""

import inspect
import os
from pathlib import Path
import subprocess
import sys

import abagen.images
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from brainspace.mesh.mesh_creation import build_polydata

from brainstat.context import genetics


def _surfaces():
    return tuple(
        build_polydata(
            np.column_stack((np.arange(n), np.zeros(n), np.ones(n))),
            np.array([[0, 1, 2]]),
        )
        for n in (3, 4)
    )


def test_import_does_not_patch_pandas_or_abagen():
    code = """
import pandas as pd
import abagen.images
before = (getattr(pd.DataFrame, 'append', None), pd.DataFrame.set_axis,
          abagen.images.labeltable_to_df)
import brainstat.context.genetics
assert before == (getattr(pd.DataFrame, 'append', None), pd.DataFrame.set_axis,
                  abagen.images.labeltable_to_df)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("failure", [None, "atlas", "expression"])
def test_geometry_split_forwarding_cleanup_and_restore(monkeypatch, failure):
    surfaces = _surfaces()
    original_points = [surface.Points.copy() for surface in surfaces]
    paths = []
    result = (pd.DataFrame({"gene": [1.0]}), {"counts": 1}, "report")
    originals = (
        getattr(pd.DataFrame, "append", None),
        pd.DataFrame.set_axis,
        abagen.images.labeltable_to_df,
    )
    sentinel = object()

    def atlas(labels, geometry, space):
        paths.extend(labels)
        paths.extend(geometry)
        assert space == "fsaverage"
        assert [len(nib.load(path).agg_data()) for path in labels] == [3, 4]
        for path in labels:
            image = nib.load(path)
            assert image.darrays[0].data.dtype == np.int32
            assert image.darrays[0].intent == 1002
        assert callable(pd.DataFrame.append)
        if failure == "atlas":
            raise RuntimeError("atlas failed")
        return sentinel

    def expression(atlas, **kwargs):
        assert atlas is sentinel
        signature = inspect.signature(genetics.surface_genetic_expression)
        expected = {
            key: value.default
            for key, value in signature.parameters.items()
            if key not in ("labels", "surfaces", "space")
        }
        expected.update(
            return_counts=True,
            return_donors=True,
            return_report=True,
            ibf_threshold=0.7,
            n_proc=2,
        )
        assert kwargs == expected
        if failure == "expression":
            raise RuntimeError("expression failed")
        return result

    monkeypatch.setattr(genetics, "check_atlas", atlas)
    monkeypatch.setattr(genetics, "get_expression_data", expression)
    kwargs = dict(
        surfaces=surfaces,
        space="fsaverage",
        return_counts=True,
        return_donors=True,
        return_report=True,
        ibf_threshold=0.7,
        n_proc=2,
    )
    if failure:
        with pytest.raises(RuntimeError):
            genetics.surface_genetic_expression(np.arange(7, dtype=np.int64), **kwargs)
    else:
        assert (
            genetics.surface_genetic_expression(np.arange(7, dtype=np.int64), **kwargs)
            is result
        )
    assert all(not Path(path).exists() for path in paths)
    assert originals == (
        getattr(pd.DataFrame, "append", None),
        pd.DataFrame.set_axis,
        abagen.images.labeltable_to_df,
    )
    for surface, points in zip(surfaces, original_points):
        np.testing.assert_array_equal(surface.Points, points)


def test_real_abagen_atlas_conversion_and_labeltable_without_background():
    labels = ({1: "left"}, {2: "right"})
    with genetics._abagen_compat():
        table = abagen.images.labeltable_to_df(labels)
        assert table.index.tolist() == [1, 2]
        assert table.hemisphere.tolist() == ["L", "R"]
        frame = pd.DataFrame({"a": [1]})
        assert frame.append(pd.DataFrame({"a": [2]}), ignore_index=True).a.tolist() == [
            1,
            2,
        ]
        assert frame.set_axis(["b"], axis=1, inplace=False).columns.tolist() == ["b"]
        assert frame.columns.tolist() == ["a"]


@pytest.mark.parametrize(
    "labels",
    [
        np.arange(6),
        np.ones((1, 7)),
        np.array([0.0, 1.5]),
        np.array([np.nan]),
        np.array([-1]),
        np.array([2**32]),
    ],
)
def test_invalid_label_arrays_fail_before_abagen(labels):
    with pytest.raises(ValueError, match="labels|Label"):
        genetics.surface_genetic_expression(labels, _surfaces(), space="fsaverage")


def test_files_forwarded_without_assuming_geometry(monkeypatch):
    sentinel = pd.DataFrame({"gene": [1.0]})

    def check(labels, geometry, space):
        assert labels == ("left.label.gii", "right.label.gii")
        assert geometry is None
        return "atlas"

    monkeypatch.setattr(genetics, "check_atlas", check)
    monkeypatch.setattr(genetics, "get_expression_data", lambda *a, **k: sentinel)
    assert (
        genetics.surface_genetic_expression(
            [Path("left.label.gii"), Path("right.label.gii")]
        )
        is sentinel
    )


def test_save_failure_cleans_created_files(monkeypatch):
    paths = []

    def fail_save(image, path):
        paths.append(Path(path))
        Path(path).write_bytes(b"partial")
        raise OSError("save failed")

    monkeypatch.setattr(genetics.nib, "save", fail_save)
    with pytest.raises(OSError, match="save failed"):
        genetics.surface_genetic_expression(np.arange(7), _surfaces())
    assert all(not path.exists() and not path.parent.exists() for path in paths)


def test_default_geometry_rejects_non_fsaverage5():
    with pytest.raises(ValueError, match="fsaverage5"):
        genetics.surface_genetic_expression(np.arange(7))


@pytest.mark.skipif(
    os.environ.get("BRAINSTAT_LIVE_TESTS") != "1",
    reason="Set BRAINSTAT_LIVE_TESTS=1 to download Allen expression data",
)
def test_surface_genetic_expression_live():
    from nilearn import datasets

    destrieux = datasets.fetch_atlas_surf_destrieux()
    labels = np.hstack((destrieux["map_left"], destrieux["map_right"]))
    fsaverage = datasets.fetch_surf_fsaverage()
    expression = genetics.surface_genetic_expression(
        labels, (fsaverage["pial_left"], fsaverage["pial_right"]), space="fsaverage"
    )
    assert isinstance(expression, pd.DataFrame)
    assert not expression.empty


def test_real_abagen_atlas_conversion(monkeypatch):
    result = pd.DataFrame({"gene": [1.0, 2.0]}, index=[1, 2])

    def expression(atlas, **kwargs):
        np.testing.assert_array_equal(atlas.labels, [1, 2])
        assert atlas.atlas_info.hemisphere.tolist() == ["L", "R"]
        return result

    monkeypatch.setattr(genetics, "get_expression_data", expression)
    actual = genetics.surface_genetic_expression(
        np.array([1, 1, 1, 2, 2, 2, 2]), _surfaces(), space="fsaverage"
    )
    assert actual is result
