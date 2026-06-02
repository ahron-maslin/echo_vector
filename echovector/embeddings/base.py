"""Base protocols and types for EchoVector embedding backends."""

from typing import Protocol

import numpy as np
import numpy.typing as npt


class EmbeddingBackend(Protocol):
    """Protocol for embedding models."""

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the generated embeddings.

        Returns:
            The integer dimension size.
        """
        ...

    def embed_audio(self, audio_paths: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of audio files.

        Args:
            audio_paths: List of file paths to audio files.

        Returns:
            A numpy array of shape (batch_size, embedding_dim).
        """
        ...

    def embed_text(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of text queries.

        Raises NotImplementedError if the backend does not support text embeddings.

        Args:
            texts: List of text strings.

        Returns:
            A numpy array of shape (batch_size, embedding_dim).
        """
        ...
