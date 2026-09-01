"""Factory for creating embedding backends."""

import importlib
from typing import Any, ClassVar

from echovector.embeddings.base import EmbeddingBackend

#: Optional extra that installs the heavy model stack. Named for the backend
#: people come here wanting, though it enables the audio-only encoders too.
MODEL_EXTRA = "clap"

#: Backends whose module needs PyTorch and transformers, which the base install
#: deliberately does not carry -- they add roughly 4 GB.
_BACKENDS_NEEDING_MODELS = frozenset({"clap", "wav2vec2", "hubert", "ast"})


class MissingBackendDependencyError(ImportError):
    """Raised when a backend's module cannot be imported for lack of a dependency."""


# Map backend names to their (module, class) for lazy import.
_BACKEND_LOCATIONS: dict[str, tuple[str, str]] = {
    "clap": ("echovector.embeddings.clap", "ClapBackend"),
    "wav2vec2": ("echovector.embeddings.transformers_audio", "Wav2Vec2Backend"),
    "hubert": ("echovector.embeddings.transformers_audio", "HubertBackend"),
    "ast": ("echovector.embeddings.transformers_audio", "ASTBackend"),
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
        cls._registry[name.lower()] = backend_cls

    @classmethod
    def available_backends(cls) -> list[str]:
        """Return every backend name that can be created.

        Returns:
            Sorted backend names, including any registered at runtime.
        """
        return sorted(set(_BACKEND_LOCATIONS) | set(cls._registry))

    @classmethod
    def resolve(cls, name: str) -> type[EmbeddingBackend]:
        """Resolve a backend name to its class without instantiating it.

        Args:
            name: The name of the backend (e.g., 'clap', 'local').

        Returns:
            The backend class.

        Raises:
            ValueError: If the backend name is not registered.
            MissingBackendDependencyError: If the backend needs a dependency the
                current install does not have.
        """
        name_lower = name.lower()

        if name_lower not in cls._registry:
            if name_lower not in _BACKEND_LOCATIONS:
                valid_names = ", ".join(cls.available_backends())
                raise ValueError(
                    f"Unknown embedding backend: '{name}'. Valid options are: {valid_names}"
                )
            module_path, class_name = _BACKEND_LOCATIONS[name_lower]
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                raise cls._missing_dependency_error(name_lower, exc) from exc
            cls._registry[name_lower] = getattr(module, class_name)

        return cls._registry[name_lower]

    @staticmethod
    def _missing_dependency_error(name: str, cause: ImportError) -> MissingBackendDependencyError:
        """Explain a failed backend import in terms of what to install.

        A bare `ModuleNotFoundError: No module named 'torch'` tells a user
        nothing about how to fix it, and the model stack is an optional extra
        precisely because most people should not download it by accident.

        Args:
            name: The backend that could not be loaded.
            cause: The underlying import failure.

        Returns:
            An error naming the exact install command.
        """
        if name in _BACKENDS_NEEDING_MODELS:
            return MissingBackendDependencyError(
                f"The '{name}' backend needs PyTorch and transformers, which are not installed. "
                f"They are an optional extra because they add roughly 4 GB:\n\n"
                f"    pip install 'echo_vector[{MODEL_EXTRA}]'\n\n"
                f"On a machine without a GPU, install the much smaller CPU build first:\n\n"
                f"    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                f"    pip install 'echo_vector[{MODEL_EXTRA}]'\n\n"
                f"The 'local' backend needs no extra, but is a smoke-test stand-in rather than "
                f"a semantic search backend. (Underlying error: {cause})"
            )
        return MissingBackendDependencyError(f"Cannot load the '{name}' backend: {cause}")

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
        return cls.resolve(name)(**kwargs)


def backend_class_for(name: str) -> type[EmbeddingBackend]:
    """Resolve a backend name to its class without constructing it.

    Lets callers inspect a backend's constructor — for example to decide
    whether a `device` argument is meaningful — before paying for a model load.

    Args:
        name: The backend name.

    Returns:
        The backend class.
    """
    return EmbeddingFactory.resolve(name)


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
