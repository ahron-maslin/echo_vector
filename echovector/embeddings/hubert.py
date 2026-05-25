"""
HuBERT embedding backend stub.
"""

from typing import List, Any
import numpy as np
import numpy.typing as npt

from echovector.embeddings.base import EmbeddingBackend


class HubertBackend(EmbeddingBackend):
    """
    Stub implementation for the HuBERT embedding backend.
    """

    def __init__(self, model_name: str = "facebook/hubert-base-ls960", **kwargs: Any) -> None:
        """Initialize the stub HuBERT backend."""
        self.model_name = model_name

    @property
    def embedding_dim(self) -> int:
        """Return a stub embedding dimension."""
        return 768

    def embed_audio(self, audio_paths: List[str]) -> npt.NDArray[np.float32]:
        """
        Embed a batch of audio files.

        Args:
            audio_paths: List of file paths to audio files.

        Raises:
            NotImplementedError: As this is a stub.
        """
        raise NotImplementedError("HuBERT audio embedding not implemented.")

    def embed_text(self, texts: List[str]) -> npt.NDArray[np.float32]:
        """
        Embed a batch of text queries.

        Args:
            texts: List of text strings.

        Raises:
            NotImplementedError: HuBERT backend does not support text embeddings.
        """
        raise NotImplementedError("HuBERT backend does not support text embeddings.")
