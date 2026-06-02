"""Base interfaces for the indexing module."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class BaseStore(ABC):
    """Abstract base class for metadata storage."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the store (e.g., create tables)."""

    @abstractmethod
    def add(
        self, int_ids: list[int], string_ids: list[str], metadata_list: list[dict[str, Any]]
    ) -> None:
        """Add metadata and ID mappings to the store.

        Args:
            int_ids: List of integer IDs assigned by the index.
            string_ids: List of original string IDs.
            metadata_list: List of metadata dictionaries.
        """

    @abstractmethod
    def get_by_int_ids(
        self, int_ids: list[int]
    ) -> tuple[list[str | None], list[dict[str, Any] | None]]:
        """Retrieve string IDs and metadata for a list of integer IDs.

        Args:
            int_ids: List of integer IDs to query.

        Returns:
            A tuple containing a list of string IDs and a list of metadata dictionaries.
        """

    @abstractmethod
    def get_max_int_id(self) -> int:
        """Get the maximum integer ID currently in the store.

        Returns:
            The maximum integer ID, or -1 if empty.
        """

    @abstractmethod
    def close(self) -> None:
        """Close any open connections."""


class BaseIndex(ABC):
    """Abstract base class for vector indices."""

    @abstractmethod
    def add(
        self,
        embeddings: npt.NDArray[np.float32],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add embeddings, IDs, and metadata to the index.

        Args:
            embeddings: A 2D numpy array of embeddings.
            ids: A list of string IDs corresponding to the embeddings.
            metadata: An optional list of metadata dictionaries.
        """

    @abstractmethod
    def search(
        self, query_embeddings: npt.NDArray[np.float32], k: int = 10
    ) -> tuple[
        npt.NDArray[np.float32],
        list[list[str | None]],
        list[list[dict[str, Any] | None]],
    ]:
        """Search for the k nearest neighbors for each query.

        Args:
            query_embeddings: A 2D numpy array of query embeddings.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A tuple of (distances, string_ids, metadata).
        """

    @abstractmethod
    def save(self, index_path: str) -> None:
        """Save the index to disk.

        Args:
            index_path: The file path to save the index to.
        """

    @abstractmethod
    def load(self, index_path: str) -> None:
        """Load the index from disk.

        Args:
            index_path: The file path to load the index from.
        """
