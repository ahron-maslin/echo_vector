"""Tests for the high-level EchoVector API."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

from echovector import EchoVector


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
