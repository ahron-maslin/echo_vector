"""Unit tests for the index manifest."""

import json

import pytest

from echovector.indexing.manifest import (
    FORMAT_VERSION,
    IndexManifest,
    IndexMismatchError,
)


def _manifest(**overrides) -> IndexManifest:
    base = {
        "format_version": FORMAT_VERSION,
        "backend": "clap:laion/clap-htsat-unfused",
        "embedding_dim": 512,
        "chunk_seconds": 10.0,
        "overlap_seconds": 1.0,
        "sample_rate": 48000,
    }
    return IndexManifest(**{**base, **overrides})


def test_round_trip(tmp_path) -> None:
    _manifest().write(tmp_path)
    assert IndexManifest.read(tmp_path) == _manifest()


def test_missing_manifest_reads_as_none(tmp_path) -> None:
    assert IndexManifest.read(tmp_path) is None


def test_write_creates_the_store_directory(tmp_path) -> None:
    target = tmp_path / "nested" / "store"
    _manifest().write(target)
    assert IndexManifest.read(target) is not None


def test_unreadable_manifest_raises(tmp_path) -> None:
    IndexManifest.path_for(tmp_path).write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexMismatchError, match="unreadable"):
        IndexManifest.read(tmp_path)


def test_future_format_version_raises(tmp_path) -> None:
    IndexManifest.path_for(tmp_path).write_text(
        json.dumps({"format_version": FORMAT_VERSION + 1}), encoding="utf-8"
    )
    with pytest.raises(IndexMismatchError, match="manifest format"):
        IndexManifest.read(tmp_path)


def test_incomplete_manifest_raises(tmp_path) -> None:
    IndexManifest.path_for(tmp_path).write_text(
        json.dumps({"format_version": FORMAT_VERSION, "backend": "clap"}), encoding="utf-8"
    )
    with pytest.raises(IndexMismatchError, match="missing fields"):
        IndexManifest.read(tmp_path)


def test_identical_settings_are_compatible(tmp_path) -> None:
    _manifest().assert_compatible_with(_manifest(), tmp_path)


def test_different_dimension_is_rejected(tmp_path) -> None:
    with pytest.raises(IndexMismatchError, match="dimensional vectors"):
        _manifest().assert_compatible_with(_manifest(embedding_dim=8), tmp_path)


def test_different_backend_is_rejected(tmp_path) -> None:
    """Two models of the same width still occupy unrelated embedding spaces."""
    with pytest.raises(IndexMismatchError, match="built with backend"):
        _manifest().assert_compatible_with(_manifest(backend="other:model"), tmp_path)


def test_chunking_differences_are_reported_not_fatal(tmp_path) -> None:
    other = _manifest(chunk_seconds=5.0, overlap_seconds=0.0, sample_rate=16000)

    _manifest().assert_compatible_with(other, tmp_path)
    differences = _manifest().chunking_differences(other)

    assert len(differences) == 3
    assert any("chunk_seconds" in d for d in differences)
    assert any("sample_rate" in d for d in differences)


def test_no_chunking_differences_when_identical() -> None:
    assert _manifest().chunking_differences(_manifest()) == []
