"""Content-addressed caching for audio embeddings."""

import hashlib
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt


class EmbeddingCache:
    """Content-addressed storage for embeddings.

    Embeddings are cached based on the hash of the file content or text.
    """

    def __init__(self, cache_dir: str | Path = ".cache/echovector") -> None:
        """Initialize the embedding cache.

        Args:
            cache_dir: Directory to store the cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_file_hash(self, filepath: str) -> str:
        """Compute SHA-256 hash of a file's contents."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _compute_text_hash(self, text: str) -> str:
        """Compute SHA-256 hash of a text string."""
        hasher = hashlib.sha256()
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    def _get_cache_path(self, hash_key: str) -> Path:
        """Get the file path for a cached embedding."""
        return self.cache_dir / f"{hash_key}.npy"

    def get_audio_embedding(self, filepath: str) -> npt.NDArray[np.float32] | None:
        """Retrieve cached embedding for an audio file.

        Args:
            filepath: Path to the audio file.

        Returns:
            The embedding array if cached, else None.
        """
        hash_key = self._compute_file_hash(filepath)
        cache_path = self._get_cache_path(hash_key)

        if cache_path.exists():
            return cast("npt.NDArray[np.float32]", np.load(cache_path))
        return None

    def put_audio_embedding(self, filepath: str, embedding: npt.NDArray[np.float32]) -> None:
        """Store embedding for an audio file in cache.

        Args:
            filepath: Path to the audio file.
            embedding: The embedding array.
        """
        hash_key = self._compute_file_hash(filepath)
        cache_path = self._get_cache_path(hash_key)
        np.save(cache_path, embedding)

    def get_text_embedding(self, text: str) -> npt.NDArray[np.float32] | None:
        """Retrieve cached embedding for a text string.

        Args:
            text: The text string.

        Returns:
            The embedding array if cached, else None.
        """
        hash_key = self._compute_text_hash(text)
        cache_path = self._get_cache_path(hash_key)

        if cache_path.exists():
            return cast("npt.NDArray[np.float32]", np.load(cache_path))
        return None

    def put_text_embedding(self, text: str, embedding: npt.NDArray[np.float32]) -> None:
        """Store embedding for a text string in cache.

        Args:
            text: The text string.
            embedding: The embedding array.
        """
        hash_key = self._compute_text_hash(text)
        cache_path = self._get_cache_path(hash_key)
        np.save(cache_path, embedding)
