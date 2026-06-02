"""AST (Audio Spectrogram Transformer) embedding backend stub."""

import numpy as np
import numpy.typing as npt

from echovector.embeddings.base import EmbeddingBackend


class ASTBackend(EmbeddingBackend):
    """Stub implementation for the AST embedding backend."""

    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593") -> None:
        """Initialize the stub AST backend."""
        self.model_name = model_name

    @property
    def embedding_dim(self) -> int:
        """Return a stub embedding dimension."""
        return 768

    def embed_audio(self, audio_paths: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of audio files.

        Args:
            audio_paths: List of file paths to audio files.

        Raises:
            NotImplementedError: As this is a stub.
        """
        raise NotImplementedError("AST audio embedding not implemented.")

    def embed_text(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of text queries.

        Args:
            texts: List of text strings.

        Raises:
            NotImplementedError: AST backend does not support text embeddings.
        """
        raise NotImplementedError("AST backend does not support text embeddings.")
