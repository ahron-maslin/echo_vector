"""
EchoVector embeddings module.
Contains backends for generating embeddings from audio and text.
"""

from echovector.embeddings.base import EmbeddingBackend
from echovector.embeddings.cache import EmbeddingCache
from echovector.embeddings.factory import EmbeddingFactory, get_embedding_model
from echovector.embeddings.clap import ClapBackend
from echovector.embeddings.whisper_enc import WhisperBackend
from echovector.embeddings.wav2vec2 import Wav2Vec2Backend
from echovector.embeddings.hubert import HubertBackend
from echovector.embeddings.ast_model import ASTBackend


__all__ = [
    "EmbeddingBackend",
    "EmbeddingCache",
    "EmbeddingFactory",
    "get_embedding_model",
    "ClapBackend",
    "WhisperBackend",
    "Wav2Vec2Backend",
    "HubertBackend",
    "ASTBackend",
]
