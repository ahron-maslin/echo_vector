"""Tests for the high-level EchoVector API."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from echovector import EchoVector


def _write_tone(path: Path, sample_rate: int = 16_000, frequency: float = 440.0) -> None:
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    sf.write(path, (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sample_rate)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"chunk_seconds": 0.0}, "chunk_seconds"),
        ({"chunk_seconds": -1.0}, "chunk_seconds"),
        ({"overlap_seconds": -1.0}, "overlap_seconds"),
        ({"chunk_seconds": 1.0, "overlap_seconds": 1.0}, "overlap_seconds"),
    ],
)
def test_echovector_rejects_invalid_chunking_params(tmp_path, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        EchoVector(store_dir=tmp_path / "index", backend="local", **kwargs)


def test_index_no_matching_files_returns_zero(tmp_path: Path) -> None:
    engine = EchoVector(store_dir=tmp_path / "index", backend="local")
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert engine.index(empty_dir) == 0


def test_index_missing_path_raises(tmp_path: Path) -> None:
    engine = EchoVector(store_dir=tmp_path / "index", backend="local")
    with pytest.raises(FileNotFoundError):
        engine.index(tmp_path / "nope.wav")


def test_index_force_reindexes_existing_file(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path)
    engine = EchoVector(
        store_dir=tmp_path / "index",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
    )

    assert engine.index(audio_path) == 1
    assert engine.index(audio_path) == 0  # already indexed, skipped
    assert engine.index(audio_path, force=True) == 1  # re-indexed
    assert engine.stats()["chunks"] == 1


def test_index_batches_across_multiple_files(tmp_path: Path) -> None:
    """Batch flushing mid-loop should not lose or duplicate chunks."""
    for i in range(3):
        _write_tone(tmp_path / f"tone-{i}.wav", frequency=220.0 + i * 100)

    engine = EchoVector(
        store_dir=tmp_path / "index",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
    )
    indexed = engine.index(tmp_path, batch_size=2)
    assert indexed == 3
    assert engine.stats()["chunks"] == 3


def test_search_rejects_negative_top_k(tmp_path: Path) -> None:
    engine = EchoVector(store_dir=tmp_path / "index", backend="local")
    with pytest.raises(ValueError, match="top_k"):
        engine.search("anything", top_k=-1)


def test_search_top_k_zero_returns_empty(tmp_path: Path) -> None:
    engine = EchoVector(store_dir=tmp_path / "index", backend="local")
    assert engine.search("anything", top_k=0) == []


def test_reset_clears_index_and_recreates_store(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path)
    engine = EchoVector(
        store_dir=tmp_path / "index",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
    )
    engine.index(audio_path)
    assert engine.stats()["chunks"] == 1
    assert engine.index_path.exists()

    engine.reset()

    assert engine.stats()["chunks"] == 0
    assert not engine.index_path.exists()

    # Store must still be usable after reset.
    assert engine.index(audio_path) == 1


def test_reset_on_fresh_engine_is_a_noop(tmp_path: Path) -> None:
    """reset() before anything is indexed should not raise."""
    engine = EchoVector(store_dir=tmp_path / "index", backend="local")
    engine.reset()
    assert engine.stats()["chunks"] == 0


def test_resolve_audio_files_and_is_indexed(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path)
    engine = EchoVector(
        store_dir=tmp_path / "index",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
    )

    resolved = engine.resolve_audio_files(tmp_path)
    assert resolved == [audio_path]
    assert engine.is_indexed(audio_path) is False

    engine.index(audio_path)
    assert engine.is_indexed(audio_path) is True


def test_echovector_indexes_audio_chunks_with_timestamps(tmp_path: Path) -> None:
    """Search should return the matching chunk, not just the whole file."""
    audio_path = tmp_path / "two_tones.wav"
    store_dir = tmp_path / "index"
    sample_rate = 16_000

    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    low = 0.25 * np.sin(2 * np.pi * 110 * t)
    high = 0.25 * np.sin(2 * np.pi * 880 * t)
    sf.write(audio_path, np.concatenate([low, high]).astype(np.float32), sample_rate)

    engine = EchoVector(
        store_dir=store_dir,
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=sample_rate,
    )

    indexed = engine.index(audio_path)
    results = engine.search("high bright tone", top_k=1)

    assert indexed == 2
    assert len(results) == 1
    assert results[0].filepath == str(audio_path)
    assert results[0].timestamp_range.start == 1.0
    assert results[0].timestamp_range.end == 2.0
    assert results[0].metadata["chunk_id"] == 1


class CountingBackend:
    """Backend that exposes whether search touches audio embedding."""

    def __init__(self) -> None:
        self.audio_calls = 0
        self.text_calls = 0

    @property
    def embedding_dim(self) -> int:
        """Return the test embedding dimension."""
        return 2

    def embed_audio(self, audio_paths: list[str]) -> npt.NDArray[np.float32]:
        """Embed audio paths for tests."""
        self.audio_calls += 1
        return np.array([[1.0, 0.0] for _ in audio_paths], dtype=np.float32)

    def embed_text(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed text queries for tests."""
        self.text_calls += 1
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_search_uses_precomputed_index_without_reprocessing_audio(tmp_path: Path) -> None:
    """Repeated text searches should not reopen or re-embed audio files."""
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    sample_rate = 16_000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    sf.write(audio_path, 0.25 * np.sin(2 * np.pi * 440 * t), sample_rate)

    backend = CountingBackend()
    engine = EchoVector(
        store_dir=store_dir,
        backend=backend,
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=sample_rate,
    )
    engine.index(audio_path)
    engine.audio_processor.load_audio = lambda _: (_ for _ in ()).throw(AssertionError)

    results = engine.search("tone", top_k=1)

    assert len(results) == 1
    assert backend.audio_calls == 1
    assert backend.text_calls == 1
