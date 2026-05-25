"""
Wav2Vec2 embedding backend stub.
"""

from typing import List, Any
import numpy as np
import numpy.typing as npt

from echovector.embeddings.base import EmbeddingBackend


class Wav2Vec2Backend(EmbeddingBackend):
    """
    Stub implementation for the Wav2Vec2 embedding backend.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base", **kwargs: Any) -> None:
        """Initialize the stub Wav2Vec2 backend."""
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
        raise NotImplementedError("Wav2Vec2 audio embedding not implemented.")

    def embed_text(self, texts: List[str]) -> npt.NDArray[np.float32]:
        """
        Embed a batch of text queries.

        Args:
            texts: List of text strings.

        Raises:
            NotImplementedError: Wav2Vec2 backend does not support text embeddings.
        """
        raise NotImplementedError("Wav2Vec2 backend does not support text embeddings.")
