"""Small signal-processing helpers, kept free of heavy dependencies.

These exist so a base install does not need librosa. librosa is excellent, but
it pulls numba and llvmlite — around 345 MB — and EchoVector only ever used it
for resampling and four spectral statistics. Resampling goes to `soxr`, which is
the library librosa itself resamples with and weighs half a megabyte, and the
statistics are short enough to write against numpy directly.

librosa remains an optional dependency for decoding formats libsndfile cannot
read (m4a/aac), and the model backends still use it.
"""

import numpy as np
import numpy.typing as npt
import soxr

__all__ = [
    "resample",
    "spectral_bandwidth",
    "spectral_centroid",
    "spectral_flatness",
    "zero_crossing_rate",
]

#: STFT geometry. Matches librosa's defaults so values stay comparable with the
#: wider literature and with anything previously computed through it.
FRAME_LENGTH = 2048
HOP_LENGTH = 512


def resample(
    audio: npt.NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
) -> npt.NDArray[np.float32]:
    """Resample a mono waveform.

    Args:
        audio: Mono float32 samples.
        orig_sr: Current sample rate.
        target_sr: Desired sample rate.

    Returns:
        The resampled waveform, or the input unchanged when the rates match.
    """
    if orig_sr == target_sr or audio.size == 0:
        return np.asarray(audio, dtype=np.float32)
    resampled = soxr.resample(
        np.asarray(audio, dtype=np.float32), orig_sr, target_sr, quality="HQ"
    )
    return np.asarray(resampled, dtype=np.float32)


def _frames(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Split a waveform into overlapping analysis frames.

    Centred and reflect-padded, matching librosa's default framing so frame
    counts and values line up.
    """
    padded = np.pad(audio, FRAME_LENGTH // 2, mode="reflect" if audio.size > 1 else "constant")
    count = 1 + (len(padded) - FRAME_LENGTH) // HOP_LENGTH
    if count < 1:
        return np.zeros((1, FRAME_LENGTH), dtype=np.float32)
    strides = (padded.strides[0] * HOP_LENGTH, padded.strides[0])
    return np.lib.stride_tricks.as_strided(padded, shape=(count, FRAME_LENGTH), strides=strides)


def _magnitude_spectrum(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return the windowed magnitude spectrogram, shaped (frames, bins)."""
    frames = _frames(audio) * np.hanning(FRAME_LENGTH).astype(np.float32)
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)


def _bin_frequencies(sample_rate: int) -> npt.NDArray[np.float32]:
    """Return the centre frequency of each FFT bin."""
    return np.fft.rfftfreq(FRAME_LENGTH, d=1.0 / sample_rate).astype(np.float32)


def spectral_centroid(audio: npt.NDArray[np.float32], sample_rate: int) -> float:
    """Return the mean spectral centroid in Hz.

    Args:
        audio: Mono float32 samples.
        sample_rate: Sample rate of `audio`.

    Returns:
        The magnitude-weighted mean frequency, averaged over frames.
    """
    magnitude = _magnitude_spectrum(audio)
    total = magnitude.sum(axis=1)
    total[total == 0.0] = 1.0
    centroids = (magnitude * _bin_frequencies(sample_rate)).sum(axis=1) / total
    return float(np.mean(centroids))


def spectral_bandwidth(audio: npt.NDArray[np.float32], sample_rate: int) -> float:
    """Return the mean spectral bandwidth in Hz.

    Args:
        audio: Mono float32 samples.
        sample_rate: Sample rate of `audio`.

    Returns:
        The magnitude-weighted spread about the centroid, averaged over frames.
    """
    magnitude = _magnitude_spectrum(audio)
    frequencies = _bin_frequencies(sample_rate)
    total = magnitude.sum(axis=1)
    total[total == 0.0] = 1.0
    centroids = (magnitude * frequencies).sum(axis=1) / total
    deviation = (frequencies[None, :] - centroids[:, None]) ** 2
    spread = np.sqrt((magnitude * deviation).sum(axis=1) / total)
    return float(np.mean(spread))


def spectral_flatness(audio: npt.NDArray[np.float32]) -> float:
    """Return the mean spectral flatness.

    Flatness contrasts the geometric and arithmetic means of the power
    spectrum: near 1 for noise, near 0 for a tone.

    Args:
        audio: Mono float32 samples.

    Returns:
        Mean flatness over frames, in [0, 1].
    """
    power = np.maximum(_magnitude_spectrum(audio) ** 2, 1e-10)
    geometric = np.exp(np.mean(np.log(power), axis=1))
    arithmetic = np.mean(power, axis=1)
    arithmetic[arithmetic == 0.0] = 1.0
    return float(np.mean(geometric / arithmetic))


def zero_crossing_rate(audio: npt.NDArray[np.float32]) -> float:
    """Return the mean zero-crossing rate.

    Args:
        audio: Mono float32 samples.

    Returns:
        The proportion of adjacent samples that change sign, averaged over frames.
    """
    frames = _frames(audio)
    # `signbit` rather than `sign` so that exact zeros do not read as a crossing
    # on both sides, which would double-count silence.
    crossings = np.abs(np.diff(np.signbit(frames).astype(np.int8), axis=1)).sum(axis=1)
    return float(np.mean(crossings / frames.shape[1]))
