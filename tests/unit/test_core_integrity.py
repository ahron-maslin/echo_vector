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

    assert stats["backend"] == "local:acoustic-v3:sr=16000"
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
    assert len(engine.search("anything", top_k=3, merge_spans=False)) == 3


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

    results = engine.search(
        "tone", top_k=2, filters=SearchFilter(min_score=-1.0), merge_spans=False
    )

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


# ---------------------------------------------------------------------------
# Checkpointed indexing
# ---------------------------------------------------------------------------


def test_indexing_checkpoints_progress_to_disk(tmp_path: Path) -> None:
    """A long run must not lose everything when it is interrupted."""
    for i in range(4):
        _write_tone(tmp_path / f"tone-{i}.wav", frequency=220.0 + i * 110)

    engine = _engine(tmp_path / "store")
    saves: list[str] = []
    original = engine.index_backend.save
    engine.index_backend.save = lambda p: (saves.append(p), original(p))[1]  # type: ignore[method-assign]

    engine.index(tmp_path, checkpoint_every_chunks=3)

    # Several mid-run saves plus the final one, rather than a single save at the end.
    assert len(saves) > 1


def test_checkpointing_leaves_the_two_halves_consistent(tmp_path: Path) -> None:
    """A checkpoint must flush pending chunks before saving, or it desyncs."""
    for i in range(4):
        _write_tone(tmp_path / f"tone-{i}.wav", frequency=220.0 + i * 110)

    engine = _engine(tmp_path / "store")
    engine.index(tmp_path, batch_size=5, checkpoint_every_chunks=3)
    engine.close()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reopened = _engine(tmp_path / "store")
    assert reopened.index_backend.desync_report() is None
    assert reopened.stats()["chunks"] == 12


def test_manifest_is_written_before_indexing_starts(tmp_path: Path) -> None:
    """An interrupted run should still leave a store that describes itself."""
    _write_tone(tmp_path / "tone.wav")
    store = tmp_path / "store"
    engine = _engine(store)

    seen: list[bool] = []
    engine.index(
        tmp_path / "tone.wav",
        on_file_indexed=lambda *_: seen.append((store / "manifest.json").exists()),
    )

    assert seen == [True]


def test_checkpointing_can_be_disabled(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav")
    engine = _engine(tmp_path / "store")
    saves: list[str] = []
    original = engine.index_backend.save
    engine.index_backend.save = lambda p: (saves.append(p), original(p))[1]  # type: ignore[method-assign]

    engine.index(tmp_path / "tone.wav", checkpoint_every_chunks=0)

    assert len(saves) == 1


# ---------------------------------------------------------------------------
# Audio-to-audio search
# ---------------------------------------------------------------------------


def test_search_by_audio_finds_acoustically_similar_files(tmp_path: Path) -> None:
    """`find more that sounds like this` needs no text tower at all."""
    _write_tone(tmp_path / "query.wav", frequency=880.0, duration=1.0)
    _write_tone(tmp_path / "similar.wav", frequency=900.0, duration=1.0)
    _write_tone(tmp_path / "different.wav", frequency=80.0, duration=1.0)

    engine = _engine(tmp_path / "store")
    engine.index(tmp_path)

    results = engine.search_by_audio(tmp_path / "query.wav", top_k=1)

    assert len(results) == 1
    assert Path(results[0].filepath).name == "similar.wav"


def test_search_by_audio_excludes_the_query_file_by_default(tmp_path: Path) -> None:
    """The query file is a perfect match for itself and would take every slot."""
    _write_tone(tmp_path / "query.wav", frequency=880.0, duration=1.0)
    _write_tone(tmp_path / "other.wav", frequency=300.0, duration=1.0)

    engine = _engine(tmp_path / "store")
    engine.index(tmp_path)
    query = tmp_path / "query.wav"

    excluded = engine.search_by_audio(query, top_k=5)
    included = engine.search_by_audio(query, top_k=5, exclude_source=False)

    assert str(query.resolve()) not in {r.filepath for r in excluded}
    assert str(query.resolve()) in {r.filepath for r in included}


def test_search_by_audio_accepts_a_waveform(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav", frequency=440.0, duration=1.0)
    engine = _engine(tmp_path / "store")
    engine.index(tmp_path / "tone.wav")

    t = np.linspace(0, 1.0, 16_000, endpoint=False)
    waveform = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    results = engine.search_by_audio(waveform, top_k=1, sample_rate=16_000)

    assert len(results) == 1


def test_search_by_audio_rejects_stereo_waveforms(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "store")
    with pytest.raises(ValueError, match="mono"):
        engine.search_by_audio(np.zeros((100, 2), dtype=np.float32), sample_rate=16_000)


def test_search_by_audio_rejects_negative_top_k(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "store")
    with pytest.raises(ValueError, match="top_k"):
        engine.search_by_audio(np.zeros(100, dtype=np.float32), top_k=-1)


def test_search_by_audio_top_k_zero_returns_empty(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "store")
    assert engine.search_by_audio(np.zeros(100, dtype=np.float32), top_k=0) == []


def test_search_by_audio_applies_filters(tmp_path: Path) -> None:
    _write_tone(tmp_path / "a.wav", frequency=440.0, duration=1.0)
    _write_tone(tmp_path / "b.wav", frequency=880.0, duration=1.0)
    engine = _engine(tmp_path / "store")
    engine.index(tmp_path)

    results = engine.search_by_audio(
        tmp_path / "a.wav", top_k=5, filters=SearchFilter(min_score=1.5)
    )

    assert results == []


def test_search_by_audio_works_with_a_path_only_backend(tmp_path: Path) -> None:
    """Backends without the waveform protocol go through a temp file."""
    _write_tone(tmp_path / "tone.wav", duration=1.0)
    engine = _engine(tmp_path / "store", backend=SameWidthBackend())
    engine.index(tmp_path / "tone.wav")

    results = engine.search_by_audio(tmp_path / "tone.wav", top_k=1, exclude_source=False)

    assert len(results) == 1


# ---------------------------------------------------------------------------
# Span merging
# ---------------------------------------------------------------------------


def test_search_returns_moments_not_chunk_slices(tmp_path: Path) -> None:
    """One long passage should be one result, not five slices of itself."""
    _write_tone(tmp_path / "long.wav", frequency=880.0, duration=8.0)
    engine = _engine(tmp_path / "store")
    engine.index(tmp_path / "long.wav")

    merged = engine.search("high bright tone", top_k=5)
    raw = engine.search("high bright tone", top_k=5, merge_spans=False)

    assert len(raw) == 5
    assert len(merged) == 1
    assert merged[0].timestamp_range.start == 0.0
    assert merged[0].timestamp_range.end == 8.0
    assert merged[0].metadata["merged_chunk_count"] == 8


def test_merging_does_not_absorb_an_irrelevant_neighbour(tmp_path: Path) -> None:
    """Adjacency alone must not widen a hit over audio the user did not ask for."""
    sample_rate = 16_000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    low = 0.25 * np.sin(2 * np.pi * 110 * t)
    high = 0.25 * np.sin(2 * np.pi * 880 * t)
    sf.write(tmp_path / "two.wav", np.concatenate([low, high]).astype(np.float32), sample_rate)

    engine = _engine(tmp_path / "store")
    engine.index(tmp_path / "two.wav")

    results = engine.search("high bright tone", top_k=1)

    assert results[0].timestamp_range.start == 1.0
    assert results[0].timestamp_range.end == 2.0


def test_merged_search_still_fills_the_page(tmp_path: Path) -> None:
    """Merging collapses candidates, so retrieval must widen to compensate."""
    for i in range(4):
        _write_tone(tmp_path / f"tone-{i}.wav", frequency=200.0 + i * 200, duration=4.0)

    engine = _engine(tmp_path / "store")
    engine.index(tmp_path)

    results = engine.search("tone", top_k=4)

    assert len(results) == 4
    assert len({r.filepath for r in results}) == 4


def test_max_per_file_diversifies_results(tmp_path: Path) -> None:
    _write_tone(tmp_path / "a.wav", frequency=880.0, duration=6.0)
    _write_tone(tmp_path / "b.wav", frequency=870.0, duration=6.0)
    engine = _engine(tmp_path / "store")
    engine.index(tmp_path)

    results = engine.search("high tone", top_k=6, merge_spans=False, max_per_file=1)

    assert len(results) == len({r.filepath for r in results})


# ---------------------------------------------------------------------------
# Change detection and pruning
# ---------------------------------------------------------------------------


def test_edited_file_is_reindexed(tmp_path: Path) -> None:
    """Matching on path alone left the index describing audio that no longer existed."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio, frequency=440.0, duration=3.0)
    engine = _engine(tmp_path / "store")
    assert engine.index(audio) == 3

    # Replace the file with different, longer content.
    _write_tone(audio, frequency=880.0, duration=5.0)
    os.utime(audio, (0, 0))

    assert engine.is_stale(audio) is True
    assert engine.index(audio) == 5
    assert engine.stats()["chunks"] == 5


def test_unchanged_file_is_still_skipped(tmp_path: Path) -> None:
    audio = tmp_path / "tone.wav"
    _write_tone(audio)
    engine = _engine(tmp_path / "store")
    engine.index(audio)

    assert engine.is_stale(audio) is False
    assert engine.index(audio) == 0


def test_reindexing_a_changed_file_does_not_duplicate_it(tmp_path: Path) -> None:
    audio = tmp_path / "tone.wav"
    _write_tone(audio, duration=3.0)
    engine = _engine(tmp_path / "store")
    engine.index(audio)

    _write_tone(audio, frequency=880.0, duration=3.0)
    os.utime(audio, (0, 0))
    engine.index(audio)

    assert engine.index_backend.store.count_files() == 1
    assert engine.stats()["chunks"] == 3
    assert engine.index_backend.desync_report() is None


def test_prune_removes_vanished_files(tmp_path: Path) -> None:
    """A deleted recording otherwise stays searchable, pointing at nothing."""
    _write_tone(tmp_path / "keep.wav", frequency=440.0)
    _write_tone(tmp_path / "gone.wav", frequency=880.0)
    engine = _engine(tmp_path / "store")
    engine.index(tmp_path)
    assert engine.index_backend.store.count_files() == 2

    (tmp_path / "gone.wav").unlink()
    removed = engine.prune()

    assert removed == [str((tmp_path / "gone.wav").resolve())]
    assert engine.index_backend.store.count_files() == 1
    assert engine.index_backend.desync_report() is None
    assert all("gone.wav" not in r.filepath for r in engine.search("tone", top_k=10))


def test_prune_is_a_noop_when_nothing_is_missing(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav")
    engine = _engine(tmp_path / "store")
    engine.index(tmp_path)

    assert engine.prune() == []
    assert engine.stats()["chunks"] == 3


def test_stale_files_lists_what_needs_reindexing(tmp_path: Path) -> None:
    _write_tone(tmp_path / "fresh.wav", frequency=440.0)
    _write_tone(tmp_path / "edited.wav", frequency=880.0)
    engine = _engine(tmp_path / "store")
    engine.index(tmp_path)

    _write_tone(tmp_path / "edited.wav", frequency=200.0, duration=5.0)
    os.utime(tmp_path / "edited.wav", (0, 0))

    assert engine.stale_files() == [str((tmp_path / "edited.wav").resolve())]


def test_missing_files_are_not_reported_as_stale(tmp_path: Path) -> None:
    """A vanished file needs pruning, not re-reading; it cannot be re-indexed."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)
    engine = _engine(tmp_path / "store")
    engine.index(audio)

    audio.unlink()

    assert engine.stale_files() == []
    assert engine.prune() == [str(audio.resolve())]


# ---------------------------------------------------------------------------
# Index structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_type", ["flat", "hnsw"])
def test_search_works_across_index_structures(tmp_path: Path, index_type: str) -> None:
    _write_tone(tmp_path / "high.wav", frequency=880.0, duration=4.0)
    _write_tone(tmp_path / "low.wav", frequency=110.0, duration=4.0)
    engine = _engine(tmp_path / "store", index_type=index_type)
    engine.index(tmp_path)

    results = engine.search("high bright tone", top_k=1)

    assert results
    assert Path(results[0].filepath).name == "high.wav"
    assert engine.stats()["index_type"] == index_type


def test_manifest_records_the_index_structure(tmp_path: Path) -> None:
    """Otherwise the manifest would claim 'flat' for an approximate store."""
    _write_tone(tmp_path / "tone.wav")
    store = tmp_path / "store"
    _engine(store, index_type="hnsw").index(tmp_path / "tone.wav")

    from echovector.indexing.manifest import IndexManifest

    assert IndexManifest.read(store).index_type == "hnsw"


def test_requesting_a_different_structure_warns_and_keeps_the_stored_one(tmp_path: Path) -> None:
    """Silently ignoring the request would misreport what the store actually is."""
    _write_tone(tmp_path / "tone.wav")
    store = tmp_path / "store"
    _engine(store, index_type="hnsw").index(tmp_path / "tone.wav")

    with pytest.warns(RuntimeWarning, match="was built as 'hnsw'"):
        reopened = _engine(store, index_type="flat")

    assert reopened.index_type == "hnsw"


# ---------------------------------------------------------------------------
# Parallel decoding
# ---------------------------------------------------------------------------


def test_decode_workers_do_not_change_the_index(tmp_path: Path) -> None:
    """Reading ahead is an optimisation; the resulting index must be identical.

    Compared against the serial path rather than a hardcoded ranking, because
    equally-scoring chunks tie-break arbitrarily and that ordering is not the
    property under test.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for i in range(6):
        _write_tone(audio_dir / f"tone-{i}.wav", frequency=200.0 + i * 120, duration=2.0)

    def index_with(workers: int) -> list[tuple[str, float, float]]:
        engine = _engine(tmp_path / f"store-{workers}")
        assert engine.index(audio_dir, decode_workers=workers) == 12
        assert engine.index_backend.desync_report() is None
        rows = engine.index_backend.store.get_by_int_ids(list(range(12)))[1]
        return sorted(
            (str(row["filepath"]), float(row["start"]), float(row["end"]))
            for row in rows
            if row is not None
        )

    serial = index_with(1)
    assert len(serial) == 12
    assert index_with(2) == serial
    assert index_with(4) == serial


def test_progress_callback_stays_in_order_when_prefetching(tmp_path: Path) -> None:
    """Files are decoded concurrently but must still be reported in order."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for i in range(8):
        _write_tone(audio_dir / f"tone-{i}.wav", frequency=200.0 + i * 90)

    seen: list[tuple[int, str]] = []
    engine = _engine(tmp_path / "store")
    engine.index(
        audio_dir,
        decode_workers=4,
        on_file_indexed=lambda done, _total, path: seen.append((done, path.name)),
    )

    assert [done for done, _ in seen] == list(range(1, 9))
    assert [name for _, name in seen] == sorted(name for _, name in seen)


def test_index_rejects_zero_decode_workers(tmp_path: Path) -> None:
    _write_tone(tmp_path / "tone.wav")
    with pytest.raises(ValueError, match="decode_workers"):
        _engine(tmp_path / "store").index(tmp_path / "tone.wav", decode_workers=0)


def test_a_failing_decode_still_propagates(tmp_path: Path, monkeypatch) -> None:
    """Errors must not be swallowed by the worker pool."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for i in range(4):
        _write_tone(audio_dir / f"tone-{i}.wav")

    engine = _engine(tmp_path / "store")
    monkeypatch.setattr(
        engine.audio_processor,
        "load_audio",
        lambda _path: (_ for _ in ()).throw(RuntimeError("decode exploded")),
    )

    with pytest.raises(RuntimeError, match="decode exploded"):
        engine.index(audio_dir, decode_workers=3)
