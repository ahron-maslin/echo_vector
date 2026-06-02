import numpy as np

from echovector.audio.chunker import SilenceAwareChunker


def test_silence_aware_chunker() -> None:
    chunker = SilenceAwareChunker(
        top_db=60.0, min_chunk_length=1.0, max_chunk_length=2.0, sample_rate=1000
    )

    # Create audio: 1.5s sound, 1s silence, 2.5s sound
    sound1 = np.ones(1500, dtype=np.float32)
    silence = np.zeros(1000, dtype=np.float32)
    sound2 = np.ones(2500, dtype=np.float32)

    audio = np.concatenate([sound1, silence, sound2])

    chunks = chunker.chunk(audio)

    # Expected chunks:
    # 1. 1500 samples (from sound1)
    # sound2 is 2500 samples > max_chunk_length (2000) -> split into 2000 and 500.
    # The 500 samples chunk is < min_chunk_length (1000), so it's discarded (unless it's the only one).

    assert len(chunks) == 2
    assert len(chunks[0]) == 1500
    assert len(chunks[1]) == 2000


def test_silence_aware_chunker_small_only() -> None:
    chunker = SilenceAwareChunker(min_chunk_length=1.0, sample_rate=1000)
    audio = np.ones(500, dtype=np.float32)

    chunks = chunker.chunk(audio)

    # Should keep it because it's the only one
    assert len(chunks) == 1
    assert len(chunks[0]) == 500


def test_silence_aware_chunker_empty() -> None:
    chunker = SilenceAwareChunker()
    audio = np.zeros(0, dtype=np.float32)

    chunks = chunker.chunk(audio)

    assert len(chunks) == 0
