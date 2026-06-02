"""Factory for creating embedding backends."""

from typing import Any, ClassVar

from echovector.embeddings.base import EmbeddingBackend

# Map backend names to their (module, class) for lazy import.
_BACKEND_LOCATIONS: dict[str, tuple[str, str]] = {
    "clap": ("echovector.embeddings.clap", "ClapBackend"),
    "whisper": ("echovector.embeddings.whisper_enc", "WhisperBackend"),
    "wav2vec2": ("echovector.embeddings.wav2vec2", "Wav2Vec2Backend"),
    "hubert": ("echovector.embeddings.hubert", "HubertBackend"),
    "ast": ("echovector.embeddings.ast_model", "ASTBackend"),
    "local": ("echovector.embeddings.local", "LocalFeatureBackend"),
    "smoke": ("echovector.embeddings.local", "LocalFeatureBackend"),
}


class EmbeddingFactory:
    """Factory to instantiate embedding backends by name."""

    _registry: ClassVar[dict[str, type[EmbeddingBackend]]] = {}

    @classmethod
    def register_backend(cls, name: str, backend_cls: type[EmbeddingBackend]) -> None:
        """Register a new backend class.

        Args:
            name: The name to register the backend under.
            backend_cls: The backend class.
        """
        cls._registry[name] = backend_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> EmbeddingBackend:
        """Create an embedding backend instance.

        Args:
            name: The name of the backend (e.g., 'clap', 'whisper').
            **kwargs: Additional keyword arguments to pass to the backend constructor.

        Returns:
            An instance of the requested EmbeddingBackend.

        Raises:
            ValueError: If the backend name is not registered.
        """
        import importlib

        name_lower = name.lower()

        if name_lower not in cls._registry:
            if name_lower not in _BACKEND_LOCATIONS:
                valid_names = ", ".join(_BACKEND_LOCATIONS.keys())
                raise ValueError(
                    f"Unknown embedding backend: '{name}'. Valid options are: {valid_names}"
                )
            module_path, class_name = _BACKEND_LOCATIONS[name_lower]
            module = importlib.import_module(module_path)
            cls._registry[name_lower] = getattr(module, class_name)

        return cls._registry[name_lower](**kwargs)


def get_embedding_model(
    model_type: str = "clap",
    **kwargs: Any,
) -> EmbeddingBackend:
    """Convenience function to get an embedding model.

    Args:
        model_type: The type of model to instantiate.
        **kwargs: Arguments to pass to the model constructor.

    Returns:
        An instantiated EmbeddingBackend.
    """
    return EmbeddingFactory.create(model_type, **kwargs)
