"""EchoVector embeddings module.

Contains backends for generating embeddings from audio and text.
"""

from echovector.embeddings.base import EmbeddingBackend
from echovector.embeddings.cache import EmbeddingCache
from echovector.embeddings.factory import EmbeddingFactory, get_embedding_model

__all__ = [
    "EmbeddingBackend",
    "EmbeddingCache",
    "EmbeddingFactory",
    "get_embedding_model",
]
