"""CLAP (Contrastive Language-Audio Pretraining) embedding backend."""

from typing import Any, cast

import librosa
import numpy as np
import numpy.typing as npt
import torch
from transformers import ClapModel, ClapProcessor

from echovector.embeddings.base import EmbeddingBackend

# CLAP's feature extractor crops anything longer than this to a random window,
# so chunks must stay at or below the limit for embeddings to be deterministic
# and to actually describe the timestamp range recorded alongside them.
DEFAULT_MAX_AUDIO_SECONDS = 10.0


class ClapBackend(EmbeddingBackend):
    """Embedding backend using the CLAP model from Hugging Face transformers.

    Supports both audio and text embeddings in the same semantic space.
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str | None = None,
    ) -> None:
        """Initialize the CLAP backend.

        Args:
            model_name: The Hugging Face model identifier.
            device: Device to run the model on (e.g., 'cpu', 'cuda').
        """
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor: Any = ClapProcessor.from_pretrained(model_name)  # pyright: ignore[reportArgumentType]
        self.model: Any = ClapModel.from_pretrained(model_name)  # pyright: ignore[reportArgumentType]
        self.model.to(self.device)
        self.model.eval()

        # Determine embedding dimension from model config
        self._embedding_dim = self.model.config.projection_dim

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the generated embeddings."""
        return cast("int", self._embedding_dim)

    @property
    def fingerprint(self) -> str:
        """Return the identity of the exact weights backing this backend."""
        return f"clap:{self.model_name}"

    def _feature_extractor_setting(self, name: str, default: float) -> float:
        """Read a numeric setting from the feature extractor, falling back to a default.

        Processors vary across transformers versions (and are mocked in tests),
        so a missing or non-numeric value must not break embedding.
        """
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        value = getattr(feature_extractor, name, None)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    @property
    def target_sample_rate(self) -> int:
        """Return the sample rate CLAP's feature extractor expects."""
        return int(self._feature_extractor_setting("sampling_rate", 48_000.0))

    @property
    def max_audio_seconds(self) -> float:
        """Return the longest segment CLAP embeds without cropping it."""
        return self._feature_extractor_setting("max_length_s", DEFAULT_MAX_AUDIO_SECONDS)

    def _load_and_resample(self, path: str, target_sr: int) -> npt.NDArray[np.float32]:
        """Load audio and resample to the target sample rate."""
        audio_array, _ = librosa.load(path, sr=target_sr, mono=True)
        return cast("npt.NDArray[np.float32]", audio_array)

    def embed_audio(self, audio_paths: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of audio files into the CLAP semantic space.

        Args:
            audio_paths: List of file paths to audio files.

        Returns:
            A numpy array of shape (batch_size, embedding_dim).
        """
        target_sr = self.target_sample_rate
        audios = [self._load_and_resample(path, target_sr) for path in audio_paths]
        return self._embed_waveforms(audios, target_sr)

    def embed_audio_arrays(
        self,
        waveforms: list[npt.NDArray[np.float32]],
        sample_rate: int,
    ) -> npt.NDArray[np.float32]:
        """Embed in-memory mono waveforms without a temp-file round trip.

        Args:
            waveforms: Mono float32 arrays.
            sample_rate: Sample rate shared by every waveform in the batch.

        Returns:
            A numpy array of shape (batch_size, embedding_dim).
        """
        target_sr = self.target_sample_rate
        if sample_rate != target_sr:
            waveforms = [
                cast(
                    "npt.NDArray[np.float32]",
                    librosa.resample(y=w, orig_sr=sample_rate, target_sr=target_sr),
                )
                for w in waveforms
            ]
        return self._embed_waveforms(waveforms, target_sr)

    def _embed_waveforms(
        self,
        waveforms: list[npt.NDArray[np.float32]],
        sample_rate: int,
    ) -> npt.NDArray[np.float32]:
        if not waveforms:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        inputs = cast("Any", self.processor)(
            audios=waveforms,
            return_tensors="pt",
            sampling_rate=sample_rate,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_features = self.model.get_audio_features(**inputs)
            # Normalize to ensure audio and text embeddings share the unit hypersphere
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        return cast(
            "npt.NDArray[np.float32]",
            audio_features.cpu().numpy().astype(np.float32),
        )

    def embed_text(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of text queries into the CLAP semantic space.

        Args:
            texts: List of text strings.

        Returns:
            A numpy array of shape (batch_size, embedding_dim).
        """
        inputs = cast("Any", self.processor)(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize to ensure audio and text embeddings share the unit hypersphere
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return cast(
            "npt.NDArray[np.float32]",
            text_features.cpu().numpy().astype(np.float32),
        )
