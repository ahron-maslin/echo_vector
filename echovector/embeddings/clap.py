"""CLAP (Contrastive Language-Audio Pretraining) embedding backend."""

from typing import Any, cast

import librosa
import numpy as np
import numpy.typing as npt
import torch
from transformers import ClapModel, ClapProcessor

from echovector.embeddings.base import EmbeddingBackend

_CLAP_AVAILABLE = True


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

    def _load_and_resample(self, path: str, target_sr: int) -> np.ndarray:
        """Load audio and resample to the target sample rate."""
        audio_array, _ = librosa.load(path, sr=target_sr, mono=True)
        return audio_array

    def embed_audio(self, audio_paths: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of audio files into the CLAP semantic space.

        Args:
            audio_paths: List of file paths to audio files.

        Returns:
            A numpy array of shape (batch_size, embedding_dim).
        """
        target_sr = 48000  # Default for CLAP
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        if feature_extractor is not None and hasattr(feature_extractor, "sampling_rate"):
            target_sr = int(getattr(feature_extractor, "sampling_rate", target_sr))

        audios = [self._load_and_resample(path, target_sr) for path in audio_paths]

        inputs = cast("Any", self.processor)(
            audios=audios,
            return_tensors="pt",
            sampling_rate=target_sr,
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
