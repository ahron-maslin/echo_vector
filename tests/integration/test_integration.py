"""Integration tests for EchoVector end-to-end workflows."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from echovector import EchoVector


def _write_tone(
    path: Path, frequency: float = 440.0, duration: float = 1.0, sr: int = 16_000
) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sf.write(path, (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sr)


@pytest.mark.integration
def test_full_round_trip(tmp_path: Path) -> None:
    """Index a file, search it, verify result points back to source."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio, frequency=880.0)

    ev = EchoVector(
        store_dir=tmp_path / "idx",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=16_000,
    )
    count = ev.index(audio)

    assert count == 1
    results = ev.search("high bright tone", top_k=1)
    assert len(results) == 1
    assert results[0].filepath == str(audio)
    assert results[0].score > 0.0


@pytest.mark.integration
def test_incremental_indexing_skips_already_indexed_files(tmp_path: Path) -> None:
    """Calling index() twice on the same file should not add duplicate chunks."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)

    ev = EchoVector(
        store_dir=tmp_path / "idx",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=16_000,
    )
    first = ev.index(audio)
    second = ev.index(audio)

    assert first == 1
    assert second == 0
    assert ev.stats()["chunks"] == first


@pytest.mark.integration
def test_force_reindex_replaces_existing(tmp_path: Path) -> None:
    """force=True should re-embed and replace, not accumulate chunks."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)

    ev = EchoVector(
        store_dir=tmp_path / "idx",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=16_000,
    )
    ev.index(audio)
    ev.index(audio, force=True)

    assert ev.stats()["chunks"] == 1


@pytest.mark.integration
def test_reset_clears_index(tmp_path: Path) -> None:
    """reset() should wipe the on-disk and in-memory index."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)

    ev = EchoVector(
        store_dir=tmp_path / "idx",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=16_000,
    )
    ev.index(audio)
    assert ev.stats()["chunks"] == 1

    ev.reset()
    assert ev.stats()["chunks"] == 0
    results = ev.search("tone", top_k=5)
    assert results == []


@pytest.mark.integration
def test_multi_file_search_returns_best_match(tmp_path: Path) -> None:
    """Of several indexed files, search should rank the best semantic match first."""
    low = tmp_path / "low.wav"
    high = tmp_path / "high.wav"
    _write_tone(low, frequency=110.0)
    _write_tone(high, frequency=880.0)

    ev = EchoVector(
        store_dir=tmp_path / "idx",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=16_000,
    )
    ev.index([low, high])

    results = ev.search("high bright treble tone", top_k=2)
    assert len(results) == 2
    assert "high.wav" in results[0].filepath


@pytest.mark.integration
def test_index_directory_recursive(tmp_path: Path) -> None:
    """Passing a directory should recursively discover all audio files."""
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_tone(tmp_path / "a.wav", frequency=440.0)
    _write_tone(sub / "b.wav", frequency=880.0)

    ev = EchoVector(
        store_dir=tmp_path / "idx",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=16_000,
    )
    count = ev.index(tmp_path / "idx" if False else tmp_path, recursive=True)

    assert count == 2


@pytest.mark.integration
def test_search_top_k_limits_results(tmp_path: Path) -> None:
    """top_k should cap the number of results returned."""
    for i in range(5):
        _write_tone(tmp_path / f"tone_{i}.wav", frequency=220.0 * (i + 1))

    ev = EchoVector(
        store_dir=tmp_path / "idx",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=16_000,
    )
    ev.index(tmp_path)

    assert len(ev.search("tone", top_k=3)) == 3
    assert len(ev.search("tone", top_k=1)) == 1
