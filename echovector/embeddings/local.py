"""Deterministic local embedding backend for offline smoke tests."""

from typing import cast

import numpy as np
import numpy.typing as npt
import soundfile as sf

from echovector.audio import dsp
from echovector.embeddings.base import EmbeddingBackend

#: Lowest frequency treated as "bright" when placing a spectral centroid on the
#: 0-1 scale. Below this, everything reads as bass.
_MIN_CENTROID_HZ = 50.0


class LocalFeatureBackend(EmbeddingBackend):
    """Small dependency-only backend that avoids model downloads.

    This backend is intended for CI, Kaggle smoke tests, and demos. It is not a
    replacement for CLAP when semantic text/audio alignment matters.
    """

    def __init__(self, sample_rate: int = 16_000) -> None:
        """Initialize the backend."""
        self.sample_rate = sample_rate

    @property
    def embedding_dim(self) -> int:
        """Return the feature vector size."""
        return 8

    @property
    def fingerprint(self) -> str:
        """Return the identity of this backend's feature definition."""
        return f"local:acoustic-v3:sr={self.sample_rate}"

    def _log_frequency_position(self, hz: float) -> float:
        """Place a frequency on a 0-1 scale by octave rather than by hertz.

        Dividing by Nyquist crushes everything audible into the bottom of the
        range: an 880 Hz tone scores 0.055 against a 16 kHz Nyquist, so the
        feature that should separate "high" from "low" barely moves while the
        derived inverse dominates. Octaves spread the same content evenly.
        """
        nyquist = self.sample_rate / 2
        if hz <= _MIN_CENTROID_HZ or nyquist <= _MIN_CENTROID_HZ:
            return 0.0
        span = np.log2(nyquist / _MIN_CENTROID_HZ)
        return float(np.clip(np.log2(hz / _MIN_CENTROID_HZ) / span, 0.0, 1.0))

    def embed_audio(self, audio_paths: list[str]) -> npt.NDArray[np.float32]:
        """Embed audio files using simple acoustic descriptors."""
        if not audio_paths:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack([self._embed_audio_file(path) for path in audio_paths]).astype(np.float32)

    def embed_audio_arrays(
        self,
        waveforms: list[npt.NDArray[np.float32]],
        sample_rate: int,
    ) -> npt.NDArray[np.float32]:
        """Embed in-memory mono waveforms using the same acoustic descriptors."""
        if not waveforms:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack(
            [self._embed_waveform(w, sample_rate) for w in waveforms],
        ).astype(np.float32)

    def embed_text(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed text queries with a keyword-weighted acoustic proxy."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack([self._embed_text(text) for text in texts]).astype(np.float32)

    def _embed_audio_file(self, path: str) -> npt.NDArray[np.float32]:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        return self._embed_waveform(audio, int(sample_rate))

    def _embed_waveform(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> npt.NDArray[np.float32]:
        if audio.ndim > 1:
            audio = np.asarray(np.mean(audio, axis=1), dtype=np.float32)
        if sample_rate != self.sample_rate:
            audio = dsp.resample(audio, int(sample_rate), self.sample_rate)

        if len(audio) == 0:
            return self._normalize(np.zeros(self.embedding_dim, dtype=np.float32))

        duration = len(audio) / self.sample_rate
        rms = float(np.sqrt(np.mean(np.square(audio))))
        zcr = dsp.zero_crossing_rate(audio)
        centroid = self._log_frequency_position(dsp.spectral_centroid(audio, self.sample_rate))
        bandwidth = self._log_frequency_position(dsp.spectral_bandwidth(audio, self.sample_rate))
        flatness = dsp.spectral_flatness(audio)

        features = np.array(
            [
                min(duration / 30.0, 1.0),
                min(rms * 4.0, 1.0),
                min(zcr * 8.0, 1.0),
                centroid,
                bandwidth,
                min(flatness * 10.0, 1.0),
                1.0 if rms < 0.01 else 0.0,
                max(1.0 - centroid, 0.0),
            ],
            dtype=np.float32,
        )
        return self._normalize(features)

    def _embed_text(self, text: str) -> npt.NDArray[np.float32]:
        lowered = text.lower()
        features = np.full(self.embedding_dim, 0.05, dtype=np.float32)

        if any(term in lowered for term in ("long", "duration", "extended")):
            features[0] = 1.0
        if any(term in lowered for term in ("loud", "strong", "alarm", "alert")):
            features[1] = 1.0
        if any(term in lowered for term in ("noisy", "buzz", "speech", "busy")):
            features[2] = 1.0
        if any(term in lowered for term in ("high", "treble", "bright", "alarm", "alert")):
            features[3] = 1.0
        if any(term in lowered for term in ("wide", "broadband", "noise")):
            features[4] = 1.0
        if any(term in lowered for term in ("flat", "noise", "static")):
            features[5] = 1.0
        if any(term in lowered for term in ("silent", "silence", "quiet")):
            features[6] = 1.0
        if any(term in lowered for term in ("low", "bass", "deep")):
            features[3] = 0.0
            features[4] = max(features[4], 0.2)
            features[7] = 1.0

        return self._normalize(features)

    def _normalize(self, vector: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return cast("npt.NDArray[np.float32]", (vector / norm).astype(np.float32))
