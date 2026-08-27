"""Regression tests for EchoVector behaviours that used to fail silently."""

import os
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from echovector import EchoVector, IndexMismatchError, SearchFilter


def _write_tone(path: Path, frequency: float = 440.0, duration: float = 3.0) -> None:
    sample_rate = 16_000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sf.write(path, (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sample_rate)


def _engine(store_dir: Path, **kwargs) -> EchoVector:
    params = {
        "backend": "local",
        "chunk_seconds": 1.0,
        "overlap_seconds": 0.0,
        "sample_rate": 16_000,
    }
    params.update(kwargs)
    return EchoVector(store_dir=store_dir, **params)


class SameWidthBackend:
    """A different model that happens to produce the same vector width."""

    embedding_dim = 8

    def embed_audio(self, audio_paths: list[str]) -> npt.NDArray[np.float32]:
        return np.ones((len(audio_paths), 8), dtype=np.float32)

    def embed_text(self, texts: list[str]) -> npt.NDArray[np.float32]:
        return np.ones((len(texts), 8), dtype=np.float32)


class BoundedWindowBackend(SameWidthBackend):
    """A backend that can only embed a fixed-length window, like CLAP."""

    max_audio_seconds = 10.0


# ---------------------------------------------------------------------------
# File identity
# ---------------------------------------------------------------------------


def test_same_file_via_different_paths_indexes_once(tmp_path: Path, monkeypatch) -> None:
    """`./audio`, `audio`, and an absolute path are one corpus, not three."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_tone(audio_dir / "tone.wav")
    monkeypatch.chdir(tmp_path)

    engine = _engine(tmp_path / "store")
    first = engine.index(str(audio_dir))

    assert first == 3
    assert engine.index("./audio") == 0
    assert engine.index("audio") == 0
    assert engine.index(str((audio_dir / "tone.wav").resolve())) == 0
    assert engine.index_backend.index.ntotal == 3


def test_is_indexed_accepts_any_equivalent_spelling(tmp_path: Path, monkeypatch) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_tone(audio_dir / "tone.wav")
    monkeypatch.chdir(tmp_path)

    engine = _engine(tmp_path / "store")
    engine.index(str(audio_dir))

    assert engine.is_indexed("./audio/tone.wav") is True
    assert engine.is_indexed(audio_dir / "tone.wav") is True


def test_resolved_paths_are_absolute(tmp_path: Path, monkeypatch) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_tone(audio_dir / "tone.wav")
    monkeypatch.chdir(tmp_path)

    resolved = _engine(tmp_path / "store").resolve_audio_files("./audio")

    assert [p.is_absolute() for p in resolved] == [True]


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_symlinked_file_is_not_indexed_twice(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_tone(audio_dir / "tone.wav")
    (audio_dir / "alias.wav").symlink_to(audio_dir / "tone.wav")

    engine = _engine(tmp_path / "store")
    engine.index(str(audio_dir))

    assert engine.index_backend.store.count_files() == 1


# ---------------------------------------------------------------------------
# Embedding-space provenance
# ---------------------------------------------------------------------------


def test_searching_with_a_different_backend_is_refused(tmp_path: Path) -> None:
    """Cosine scores across unrelated embedding spaces look plausible and mean nothing."""
    _write_tone(tmp_path / "tone.wav")
    store = tmp_path / "store"
    _engine(store).index(tmp_path / "tone.wav")

    reopened = _engine(store, backend=SameWidthBackend())
    with pytest.raises(IndexMismatchError, match="built with backend"):
        reopened.search("anything")


def test_indexing_with_a_different_backend_is_refused(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav")
    store = tmp_path / "store"
    _engine(store).index(tmp_path / "tone.wav")

    reopened = _engine(store, backend=SameWidthBackend())
    with pytest.raises(IndexMismatchError):
        reopened.index(tmp_path / "tone.wav", force=True)


def test_manifest_records_how_the_index_was_built(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav")
    store = tmp_path / "store"
    engine = _engine(store, chunk_seconds=2.0, overlap_seconds=0.5)
    engine.index(tmp_path / "tone.wav")

    stats = _engine(store, chunk_seconds=2.0, overlap_seconds=0.5).stats()

    assert stats["backend"] == "local:acoustic-v2:sr=16000"
    assert stats["chunk_seconds"] == 2.0
    assert stats["overlap_seconds"] == 0.5
    assert stats["sample_rate"] == 16_000
    assert stats["files"] == 1


def test_indexing_with_different_chunking_warns(tmp_path: Path) -> None:
    """Chunking drift is survivable, so it warns rather than refusing."""
    _write_tone(tmp_path / "a.wav")
    _write_tone(tmp_path / "b.wav", frequency=880.0)
    store = tmp_path / "store"
    _engine(store, chunk_seconds=1.0).index(tmp_path / "a.wav")

    with pytest.warns(RuntimeWarning, match="different chunking settings"):
        _engine(store, chunk_seconds=2.0).index(tmp_path / "b.wav")


def test_searching_a_store_never_warns_about_chunking(tmp_path: Path) -> None:
    """Chunking only affects new audio, so reading must stay quiet."""
    _write_tone(tmp_path / "tone.wav")
    store = tmp_path / "store"
    _engine(store, chunk_seconds=1.0).index(tmp_path / "tone.wav")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        engine = _engine(store, chunk_seconds=2.0)
        assert engine.search("tone", top_k=1)
        assert engine.stats()["chunks"] == 3


def test_reset_clears_the_manifest(tmp_path: Path) -> None:
    """After a reset the store must be re-describable by a different backend."""
    _write_tone(tmp_path / "tone.wav")
    store = tmp_path / "store"
    _engine(store).index(tmp_path / "tone.wav")

    engine = _engine(store)
    engine.reset()

    assert engine.stats()["backend"] == "<not yet indexed>"
    assert engine.stats()["chunks"] == 0
    assert engine.index(tmp_path / "tone.wav") == 3


def test_desync_between_index_and_store_warns(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav")
    store = tmp_path / "store"
    engine = _engine(store)
    engine.index(tmp_path / "tone.wav")
    engine.close()

    (store / "metadata.sqlite").unlink()

    with pytest.warns(RuntimeWarning, match="out of sync"):
        _engine(store)


# ---------------------------------------------------------------------------
# Chunk length vs. the backend's window
# ---------------------------------------------------------------------------


def test_chunks_longer_than_the_backend_window_are_refused(tmp_path: Path) -> None:
    """A cropped embedding would not describe the timestamps stored beside it."""
    _write_tone(tmp_path / "tone.wav")
    engine = _engine(
        tmp_path / "store",
        backend=BoundedWindowBackend(),
        chunk_seconds=30.0,
        overlap_seconds=1.0,
    )

    with pytest.raises(ValueError, match=r"exceeds the 10\.0s window"):
        engine.index(tmp_path / "tone.wav")


def test_chunks_within_the_backend_window_are_allowed(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav")
    engine = _engine(
        tmp_path / "store",
        backend=BoundedWindowBackend(),
        chunk_seconds=10.0,
        overlap_seconds=1.0,
    )

    assert engine.index(tmp_path / "tone.wav") == 1


def test_backends_without_a_window_accept_any_chunk_length(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav")
    engine = _engine(tmp_path / "store", chunk_seconds=30.0, overlap_seconds=1.0)
    assert engine.index(tmp_path / "tone.wav") == 1


# ---------------------------------------------------------------------------
# Indexing efficiency
# ---------------------------------------------------------------------------


def test_file_metadata_is_read_once_per_file(tmp_path: Path, monkeypatch) -> None:
    """Metadata describes the file; reopening it for every chunk is pure waste."""
    import echovector.core as core

    _write_tone(tmp_path / "tone.wav", duration=5.0)
    calls: list[str] = []
    original = core.extract_metadata
    monkeypatch.setattr(
        core, "extract_metadata", lambda path: (calls.append(path), original(path))[1]
    )

    chunks = _engine(tmp_path / "store").index(tmp_path / "tone.wav")

    assert chunks == 5
    assert len(calls) == 1


def test_waveform_backend_writes_no_temp_files(tmp_path: Path, monkeypatch) -> None:
    """In-memory chunks must not round-trip through disk to reach the embedder."""
    import echovector.core as core

    _write_tone(tmp_path / "tone.wav")
    monkeypatch.setattr(core.sf, "write", lambda *a, **k: pytest.fail("chunk was written to disk"))

    assert _engine(tmp_path / "store").index(tmp_path / "tone.wav") == 3


def test_path_only_backend_still_works(tmp_path: Path) -> None:
    """Backends predating the waveform protocol keep working via temp files."""
    _write_tone(tmp_path / "tone.wav")
    engine = _engine(tmp_path / "store", backend=SameWidthBackend())

    assert engine.index(tmp_path / "tone.wav") == 3
    assert len(engine.search("anything", top_k=3)) == 3


def test_index_rejects_non_positive_batch_size(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav")
    with pytest.raises(ValueError, match="batch_size"):
        _engine(tmp_path / "store").index(tmp_path / "tone.wav", batch_size=0)


def test_invalid_sample_rate_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        _engine(tmp_path / "store", sample_rate=0)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_search_applies_min_score_filter(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav")
    engine = _engine(tmp_path / "store")
    engine.index(tmp_path / "tone.wav")

    assert engine.search("high bright tone", top_k=5)
    assert engine.search("high bright tone", top_k=5, filters=SearchFilter(min_score=1.5)) == []


def test_search_applies_metadata_filter(tmp_path: Path) -> None:
    _write_tone(tmp_path / "a.wav", frequency=220.0)
    _write_tone(tmp_path / "b.wav", frequency=880.0)
    engine = _engine(tmp_path / "store")
    engine.index(tmp_path)

    results = engine.search(
        "tone", top_k=10, filters=SearchFilter(metadata_filters={"filename": "b.wav"})
    )

    assert results
    assert {r.metadata["filename"] for r in results} == {"b.wav"}


def test_search_applies_filepath_filter(tmp_path: Path) -> None:
    _write_tone(tmp_path / "a.wav", frequency=220.0)
    _write_tone(tmp_path / "b.wav", frequency=880.0)
    engine = _engine(tmp_path / "store")
    engine.index(tmp_path)
    wanted = str((tmp_path / "a.wav").resolve())

    results = engine.search("tone", top_k=10, filters=SearchFilter(filepaths=[wanted]))

    assert results
    assert {r.filepath for r in results} == {wanted}


def test_filtered_search_still_honours_top_k(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav", duration=10.0)
    engine = _engine(tmp_path / "store")
    engine.index(tmp_path / "tone.wav")

    results = engine.search("tone", top_k=2, filters=SearchFilter(min_score=-1.0))

    assert len(results) == 2


# ---------------------------------------------------------------------------
# Resource handling
# ---------------------------------------------------------------------------


def test_engine_is_a_context_manager(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav")
    with _engine(tmp_path / "store") as engine:
        assert engine.index(tmp_path / "tone.wav") == 3

    import sqlite3

    with pytest.raises(sqlite3.ProgrammingError):
        engine.index_backend.store.count()


def test_stats_does_not_construct_the_backend(tmp_path: Path) -> None:
    """A row count should never cost a model download."""
    _write_tone(tmp_path / "tone.wav")
    store = tmp_path / "store"
    _engine(store).index(tmp_path / "tone.wav")

    reopened = EchoVector(store_dir=store, backend="a-backend-that-does-not-exist")
    stats = reopened.stats()

    assert stats["chunks"] == 3
    assert stats["embedding_dim"] == 8
