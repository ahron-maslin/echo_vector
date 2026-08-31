"""Base protocols and types for EchoVector embedding backends."""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


class TextSearchUnsupportedError(NotImplementedError):
    """Raised when a text query is issued against an audio-only backend.

    Defined here rather than beside the backends that raise it so that catching
    it costs nothing: this module imports only numpy, while the audio-only
    backends pull in torch.
    """


class EmbeddingBackend(Protocol):
    """Protocol for embedding models."""

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the generated embeddings.

        Returns:
            The integer dimension size.
        """
        ...

    def embed_audio(self, audio_paths: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of audio files.

        Args:
            audio_paths: List of file paths to audio files.

        Returns:
            A numpy array of shape (batch_size, embedding_dim).
        """
        ...

    def embed_text(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of text queries.

        Raises NotImplementedError if the backend does not support text embeddings.

        Args:
            texts: List of text strings.

        Returns:
            A numpy array of shape (batch_size, embedding_dim).
        """
        ...


@runtime_checkable
class WaveformEmbeddingBackend(Protocol):
    """Backends that can embed in-memory waveforms directly.

    Indexing slices audio into chunks that already live in memory. A backend
    implementing this protocol lets :class:`~echovector.core.EchoVector` hand
    those slices over as arrays, avoiding a write-to-temp-wav/re-decode round
    trip per chunk. Backends that only implement :meth:`embed_audio` still work;
    the indexer falls back to the temp-file path for them.
    """

    def embed_audio_arrays(
        self,
        waveforms: list[npt.NDArray[np.float32]],
        sample_rate: int,
    ) -> npt.NDArray[np.float32]:
        """Embed a batch of mono waveforms.

        Args:
            waveforms: Mono float32 arrays. Lengths may differ between entries.
            sample_rate: Sample rate shared by every waveform in the batch.

        Returns:
            A numpy array of shape (batch_size, embedding_dim).
        """
        ...


@runtime_checkable
class FingerprintedBackend(Protocol):
    """Backends that can identify the exact model that produced an embedding.

    The fingerprint is persisted in the index manifest and compared when an
    existing index is reopened, so a store built with one model cannot be
    silently queried with another.
    """

    @property
    def fingerprint(self) -> str:
        """Return a stable identifier for this backend and its weights."""
        ...


def backend_fingerprint(backend: object) -> str:
    """Return the persisted identity of an embedding backend.

    Falls back to the class name for backends that do not declare a
    fingerprint, which is still enough to catch a swap between two different
    backend implementations that happen to share an embedding dimension.

    Args:
        backend: An embedding backend instance.

    Returns:
        A stable string identifying the backend.
    """
    if isinstance(backend, FingerprintedBackend):
        return backend.fingerprint
    return type(backend).__name__


def max_audio_seconds(backend: object) -> float | None:
    """Return the longest audio segment a backend can embed without truncation.

    Args:
        backend: An embedding backend instance.

    Returns:
        The limit in seconds, or None if the backend imposes no fixed limit.
    """
    limit = getattr(backend, "max_audio_seconds", None)
    return float(limit) if limit is not None else None
