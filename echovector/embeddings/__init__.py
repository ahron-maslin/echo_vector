"""EchoVector embeddings module.

Contains backends for generating embeddings from audio and text.
"""

from echovector.embeddings.base import (
    EmbeddingBackend,
    WaveformEmbeddingBackend,
    backend_fingerprint,
)
from echovector.embeddings.factory import EmbeddingFactory, get_embedding_model

__all__ = [
    "EmbeddingBackend",
    "EmbeddingFactory",
    "WaveformEmbeddingBackend",
    "backend_fingerprint",
    "get_embedding_model",
]
