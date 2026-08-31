import numpy as np
import pytest

from echovector.audio.chunker import SilenceAwareChunker

SR = 1000


def _tone(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.float32)


def _silence(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


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


# --------------------------------------------------------------------------
# chunk_with_timestamps
# --------------------------------------------------------------------------


def test_timestamps_are_absolute_across_dropped_silence() -> None:
    chunker = SilenceAwareChunker(min_chunk_length=0.5, max_chunk_length=10.0, sample_rate=SR)
    # 1s tone, 2s silence, 1.5s tone, 3s silence, 1s tone
    audio = np.concatenate(
        [
            _tone(1000),
            _silence(2000),
            _tone(1500),
            _silence(3000),
            _tone(1000),
        ]
    )

    result = chunker.chunk_with_timestamps(audio)

    spans = [(round(s, 6), round(e, 6)) for s, e, _ in result]
    assert spans == [(0.0, 1.0), (3.0, 4.5), (7.5, 8.5)]
    # The third chunk must not be reported at 4.5s-5.5s, which is where it would
    # land if offsets accumulated emitted lengths instead of source positions.
    for start, end, waveform in result:
        assert len(waveform) == pytest.approx((end - start) * SR, abs=1)
        np.testing.assert_array_equal(waveform, audio[round(start * SR) : round(end * SR)])


def test_timestamps_survive_leading_and_trailing_silence() -> None:
    chunker = SilenceAwareChunker(min_chunk_length=0.5, max_chunk_length=10.0, sample_rate=SR)
    audio = np.concatenate([_silence(2500), _tone(1000), _silence(4000)])

    result = chunker.chunk_with_timestamps(audio)

    assert len(result) == 1
    start, end, waveform = result[0]
    assert start == pytest.approx(2.5)
    assert end == pytest.approx(3.5)
    assert len(waveform) == 1000


def test_timestamps_with_no_silence_at_all() -> None:
    chunker = SilenceAwareChunker(min_chunk_length=0.5, max_chunk_length=2.0, sample_rate=SR)
    audio = _tone(5000)

    result = chunker.chunk_with_timestamps(audio)

    assert [(s, e) for s, e, _ in result] == [(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)]
    assert sum(len(w) for _, _, w in result) == 5000


def test_split_pieces_of_one_interval_are_contiguous() -> None:
    chunker = SilenceAwareChunker(min_chunk_length=0.5, max_chunk_length=1.0, sample_rate=SR)
    audio = np.concatenate([_silence(500), _tone(3000)])

    result = chunker.chunk_with_timestamps(audio)

    starts = [s for s, _, _ in result]
    ends = [e for _, e, _ in result]
    assert starts == pytest.approx([0.5, 1.5, 2.5])
    assert starts[1:] == pytest.approx(ends[:-1])
    assert ends[-1] == pytest.approx(3.5)


def test_max_chunk_length_is_never_exceeded() -> None:
    chunker = SilenceAwareChunker(min_chunk_length=0.1, max_chunk_length=1.0, sample_rate=SR)
    audio = np.concatenate([_tone(2500), _silence(1000), _tone(4100)])

    result = chunker.chunk_with_timestamps(audio)

    assert result
    for start, end, waveform in result:
        assert end - start <= 1.0 + 1e-9
        assert len(waveform) <= 1000


def test_zero_max_chunk_length_terminates() -> None:
    chunker = SilenceAwareChunker(min_chunk_length=0.0, max_chunk_length=0.0, sample_rate=SR)

    result = chunker.chunk_with_timestamps(_tone(5))

    assert len(result) == 5
    assert all(len(w) == 1 for _, _, w in result)


def test_short_fragment_is_dropped_not_merged() -> None:
    chunker = SilenceAwareChunker(min_chunk_length=1.0, max_chunk_length=10.0, sample_rate=SR)
    audio = np.concatenate([_tone(2000), _silence(2000), _tone(300)])

    result = chunker.chunk_with_timestamps(audio)

    # The 0.3s fragment is below min_chunk_length and is separated from its
    # neighbour by discarded silence, so merging would misreport the span.
    assert [(s, e) for s, e, _ in result] == [(0.0, 2.0)]


def test_lone_short_fragment_is_kept() -> None:
    chunker = SilenceAwareChunker(min_chunk_length=1.0, max_chunk_length=10.0, sample_rate=SR)
    audio = np.concatenate([_silence(1000), _tone(300)])

    result = chunker.chunk_with_timestamps(audio)

    assert len(result) == 1
    start, end, waveform = result[0]
    assert start == pytest.approx(1.0)
    assert end == pytest.approx(1.3)
    assert len(waveform) == 300


def test_empty_audio_yields_no_timestamped_chunks() -> None:
    chunker = SilenceAwareChunker(sample_rate=SR)

    assert chunker.chunk_with_timestamps(np.zeros(0, dtype=np.float32)) == []


def test_all_silence_yields_no_timestamped_chunks() -> None:
    chunker = SilenceAwareChunker(sample_rate=SR)

    assert chunker.chunk_with_timestamps(_silence(5000)) == []


def test_quiet_region_counts_as_silence() -> None:
    chunker = SilenceAwareChunker(
        top_db=60.0, min_chunk_length=0.5, max_chunk_length=10.0, sample_rate=SR
    )
    # The threshold is relative to the loudest sample, so 5e-4 against a peak of
    # 1.0 sits below the -60 dB cut and splits the two loud regions.
    quiet = np.full(2000, 5e-4, dtype=np.float32)
    audio = np.concatenate([_tone(1000), quiet, _tone(1000)])

    result = chunker.chunk_with_timestamps(audio)

    assert [(s, e) for s, e, _ in result] == [(0.0, 1.0), (3.0, 4.0)]


def test_sample_rate_argument_overrides_instance() -> None:
    chunker = SilenceAwareChunker(min_chunk_length=0.0, max_chunk_length=10.0, sample_rate=SR)
    audio = np.concatenate([_silence(4000), _tone(4000)])

    result = chunker.chunk_with_timestamps(audio, sample_rate=8000)

    assert [(s, e) for s, e, _ in result] == [(0.5, 1.0)]


def test_invalid_sample_rate_rejected() -> None:
    chunker = SilenceAwareChunker(sample_rate=SR)

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        chunker.chunk_with_timestamps(_tone(100), sample_rate=0)


def test_chunk_and_chunk_with_timestamps_agree() -> None:
    chunker = SilenceAwareChunker(min_chunk_length=0.5, max_chunk_length=1.5, sample_rate=SR)
    audio = np.concatenate([_tone(2200), _silence(800), _tone(1700), _silence(600), _tone(400)])

    bare = chunker.chunk(audio)
    stamped = chunker.chunk_with_timestamps(audio)

    assert len(bare) == len(stamped)
    for plain, (_, _, waveform) in zip(bare, stamped, strict=True):
        np.testing.assert_array_equal(plain, waveform)
