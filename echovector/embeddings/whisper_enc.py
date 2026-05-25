"""
Whisper encoder embedding backend stub.
"""

from typing import List, Any
import numpy as np
import numpy.typing as npt

from echovector.embeddings.base import EmbeddingBackend


class WhisperBackend(EmbeddingBackend):
    """
    Stub implementation for the Whisper embedding backend.
    Uses the encoder of Whisper models to extract audio features.
    """

    def __init__(self, model_name: str = "openai/whisper-base", **kwargs: Any) -> None:
        """Initialize the stub Whisper backend."""
        self.model_name = model_name

    @property
    def embedding_dim(self) -> int:
        """Return a stub embedding dimension."""
        return 512

    def embed_audio(self, audio_paths: List[str]) -> npt.NDArray[np.float32]:
        """
        Embed a batch of audio files.

        Args:
            audio_paths: List of file paths to audio files.

        Raises:
            NotImplementedError: As this is a stub.
        """
        raise NotImplementedError("Whisper audio embedding not implemented.")

    def embed_text(self, texts: List[str]) -> npt.NDArray[np.float32]:
        """
        Embed a batch of text queries.

        Args:
            texts: List of text strings.

        Raises:
            NotImplementedError: Whisper backend does not support text embeddings.
        """
        raise NotImplementedError("Whisper backend does not support text embeddings.")
