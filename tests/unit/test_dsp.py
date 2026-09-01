"""Tests for the numpy/soxr replacements for librosa.

These exist to justify dropping a 345 MB dependency, so the important ones
compare directly against librosa and are skipped when it is not installed.
"""

import numpy as np
import pytest

from echovector.audio import dsp

librosa = pytest.importorskip("librosa")

SAMPLE_RATE = 22050


def _tone(frequency: float, duration: float = 2.0, rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _noise(duration: float = 2.0, rate: int = SAMPLE_RATE) -> np.ndarray:
    return (0.2 * np.random.default_rng(0).standard_normal(int(rate * duration))).astype(
        np.float32
    )


class TestAgainstLibrosa:
    """Replacements must track librosa wherever the value is meaningful.

    Tolerances are on absolute frequency for the spectral statistics: a pure
    tone's true bandwidth is near zero, so a relative comparison there measures
    windowing leakage rather than agreement.
    """

    @pytest.mark.parametrize("frequency", [220.0, 440.0, 4000.0])
    def test_spectral_centroid_matches(self, frequency: float) -> None:
        signal = _tone(frequency)
        ours = dsp.spectral_centroid(signal, SAMPLE_RATE)
        theirs = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=SAMPLE_RATE)[0]))
        assert ours == pytest.approx(theirs, rel=0.02)

    def test_spectral_centroid_matches_on_noise(self) -> None:
        signal = _noise()
        ours = dsp.spectral_centroid(signal, SAMPLE_RATE)
        theirs = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=SAMPLE_RATE)[0]))
        assert ours == pytest.approx(theirs, rel=0.01)

    def test_spectral_bandwidth_matches_on_noise(self) -> None:
        signal = _noise()
        ours = dsp.spectral_bandwidth(signal, SAMPLE_RATE)
        theirs = float(np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=SAMPLE_RATE)[0]))
        assert ours == pytest.approx(theirs, rel=0.01)

    def test_spectral_bandwidth_stays_small_for_a_pure_tone(self) -> None:
        """Both are near zero here; agreement is in absolute Hz, not ratio."""
        signal = _tone(440.0)
        ours = dsp.spectral_bandwidth(signal, SAMPLE_RATE)
        theirs = float(np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=SAMPLE_RATE)[0]))
        assert abs(ours - theirs) < 25.0

    def test_spectral_flatness_matches_on_noise(self) -> None:
        signal = _noise()
        ours = dsp.spectral_flatness(signal)
        theirs = float(np.mean(librosa.feature.spectral_flatness(y=signal)[0]))
        assert ours == pytest.approx(theirs, rel=0.05)

    @pytest.mark.parametrize("frequency", [220.0, 4000.0])
    def test_zero_crossing_rate_matches(self, frequency: float) -> None:
        signal = _tone(frequency)
        ours = dsp.zero_crossing_rate(signal)
        theirs = float(np.mean(librosa.feature.zero_crossing_rate(y=signal)[0]))
        assert ours == pytest.approx(theirs, abs=0.01)

    @pytest.mark.parametrize(("orig", "target"), [(48000, 16000), (16000, 48000), (44100, 22050)])
    def test_resampling_is_identical(self, orig: int, target: int) -> None:
        """librosa resamples with soxr too, so this should agree exactly."""
        signal = _tone(440.0, duration=1.0, rate=orig)
        ours = dsp.resample(signal, orig, target)
        theirs = librosa.resample(y=signal, orig_sr=orig, target_sr=target)

        assert len(ours) == len(theirs)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)


class TestBehaviour:
    def test_flatness_separates_noise_from_a_tone(self) -> None:
        assert dsp.spectral_flatness(_noise()) > 0.3
        assert dsp.spectral_flatness(_tone(440.0)) < 0.01

    def test_centroid_orders_tones_by_pitch(self) -> None:
        low = dsp.spectral_centroid(_tone(200.0), SAMPLE_RATE)
        high = dsp.spectral_centroid(_tone(4000.0), SAMPLE_RATE)
        assert high > low

    def test_silence_is_handled(self) -> None:
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        assert dsp.spectral_centroid(silence, SAMPLE_RATE) == 0.0
        assert dsp.spectral_bandwidth(silence, SAMPLE_RATE) == 0.0
        assert dsp.zero_crossing_rate(silence) == 0.0
        assert dsp.spectral_flatness(silence) == pytest.approx(1.0)

    @pytest.mark.parametrize("length", [0, 1, 10, 2047])
    def test_short_and_empty_input_does_not_crash(self, length: int) -> None:
        signal = np.zeros(length, dtype=np.float32)
        assert np.isfinite(dsp.spectral_centroid(signal, SAMPLE_RATE))
        assert np.isfinite(dsp.spectral_bandwidth(signal, SAMPLE_RATE))
        assert np.isfinite(dsp.zero_crossing_rate(signal))
        assert np.isfinite(dsp.spectral_flatness(signal))

    def test_resample_is_a_noop_at_the_same_rate(self) -> None:
        signal = _tone(440.0)
        np.testing.assert_array_equal(dsp.resample(signal, SAMPLE_RATE, SAMPLE_RATE), signal)

    def test_resample_of_empty_audio(self) -> None:
        assert dsp.resample(np.zeros(0, dtype=np.float32), 48000, 16000).size == 0
