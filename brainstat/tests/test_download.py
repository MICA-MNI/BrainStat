"""Offline regressions for atomic cache publication."""

import io
from http.client import RemoteDisconnected
from urllib.error import URLError

import pytest

from brainstat._utils import _download_file


class InterruptedStream(io.BytesIO):
    def read(self, size=-1):
        if self.tell():
            raise RemoteDisconnected("interrupted response")
        return super().read(2)


@pytest.mark.parametrize("existing", [False, True])
def test_failed_download_preserves_destination_and_cleans_staging(
    tmp_path, monkeypatch, existing
):
    destination = tmp_path / "data.bin"
    if existing:
        destination.write_bytes(b"previous valid file")
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: InterruptedStream(b"partial")
    )
    monkeypatch.setattr("time.sleep", lambda *a: None)
    with pytest.raises(RemoteDisconnected):
        _download_file("https://example.invalid/data", destination, overwrite=True)
    assert destination.exists() == existing
    if existing:
        assert destination.read_bytes() == b"previous valid file"
    assert list(tmp_path.iterdir()) == ([destination] if existing else [])


def test_retry_then_publish(tmp_path, monkeypatch):
    destination = tmp_path / "data.bin"
    destination.write_bytes(b"old")
    responses = iter([URLError("unavailable"), io.BytesIO(b"complete")])

    def open_response(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("urllib.request.urlopen", open_response)
    monkeypatch.setattr("time.sleep", lambda *a: None)
    _download_file("https://example.invalid/data", destination, overwrite=True)
    assert destination.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [destination]


def test_cached_file_does_not_download(tmp_path, monkeypatch):
    destination = tmp_path / "data.bin"
    destination.write_bytes(b"cached")

    def unexpected(*args, **kwargs):
        pytest.fail("cached file should not trigger network access")

    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    _download_file("https://example.invalid/data", destination)
    assert destination.read_bytes() == b"cached"
