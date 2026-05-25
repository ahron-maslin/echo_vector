"""
Factory for creating embedding backends.
"""

from typing import Dict, Any, Type
from echovector.embeddings.base import EmbeddingBackend
from echovector.embeddings.clap import ClapBackend
from echovector.embeddings.whisper_enc import WhisperBackend
from echovector.embeddings.wav2vec2 import Wav2Vec2Backend
from echovector.embeddings.hubert import HubertBackend
from echovector.embeddings.ast_model import ASTBackend


class EmbeddingFactory:
    """Factory to instantiate embedding backends by name."""

    _registry: Dict[str, Type[EmbeddingBackend]] = {
        "clap": ClapBackend,
        "whisper": WhisperBackend,
        "wav2vec2": Wav2Vec2Backend,
        "hubert": HubertBackend,
        "ast": ASTBackend,
    }

    @classmethod
    def register_backend(cls, name: str, backend_cls: Type[EmbeddingBackend]) -> None:
        """
        Register a new backend class.

        Args:
            name: The name to register the backend under.
            backend_cls: The backend class.
        """
        cls._registry[name] = backend_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> EmbeddingBackend:
        """
        Create an embedding backend instance.

        Args:
            name: The name of the backend (e.g., 'clap', 'whisper').
            **kwargs: Additional keyword arguments to pass to the backend constructor.

        Returns:
            An instance of the requested EmbeddingBackend.

        Raises:
            ValueError: If the backend name is not registered.
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            valid_names = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown embedding backend: '{name}'. Valid options are: {valid_names}"
            )
            
        return cls._registry[name_lower](**kwargs)


def get_embedding_model(model_type: str = "clap", **kwargs: Any) -> EmbeddingBackend:
    """
    Convenience function to get an embedding model.

    Args:
        model_type: The type of model to instantiate.
        **kwargs: Arguments to pass to the model constructor.

    Returns:
        An instantiated EmbeddingBackend.
    """
    return EmbeddingFactory.create(model_type, **kwargs)
