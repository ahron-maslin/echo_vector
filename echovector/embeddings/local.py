"""Deterministic local embedding backend for offline smoke tests."""

from typing import cast

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf

from echovector.embeddings.base import EmbeddingBackend


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

    def embed_audio(self, audio_paths: list[str]) -> npt.NDArray[np.float32]:
        """Embed audio files using simple acoustic descriptors."""
        return np.vstack([self._embed_audio_file(path) for path in audio_paths]).astype(np.float32)

    def embed_text(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed text queries with a keyword-weighted acoustic proxy."""
        return np.vstack([self._embed_text(text) for text in texts]).astype(np.float32)

    def _embed_audio_file(self, path: str) -> npt.NDArray[np.float32]:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=int(sample_rate),
                target_sr=self.sample_rate,
                axis=0,
            )

        if len(audio) == 0:
            return self._normalize(np.zeros(self.embedding_dim, dtype=np.float32))

        duration = len(audio) / self.sample_rate
        rms = float(np.sqrt(np.mean(np.square(audio))))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)[0]))
        centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0])
            / (self.sample_rate / 2)
        )
        bandwidth = float(
            np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0])
            / (self.sample_rate / 2)
        )
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)[0]))

        features = np.array(
            [
                min(duration / 30.0, 1.0),
                min(rms * 4.0, 1.0),
                min(zcr * 8.0, 1.0),
                min(centroid, 1.0),
                min(bandwidth, 1.0),
                min(flatness * 10.0, 1.0),
                1.0 if rms < 0.01 else 0.0,
                max(1.0 - min(centroid, 1.0), 0.0),
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
