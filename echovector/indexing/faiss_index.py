"""Faiss-based index implementation."""

import os
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import numpy.typing as npt

from .base import BaseIndex
from .store import SQLiteStore


class FaissIndex(BaseIndex):
    """Faiss-based index using IndexFlatIP (Inner Product) for vector search."""

    def __init__(self, dimension: int, db_path: str = ":memory:") -> None:
        """Initialize the Faiss index and the metadata store.

        Args:
            dimension: Dimensionality of the embeddings.
            db_path: Path to the SQLite store database.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.store = SQLiteStore(db_path)

    def add(
        self,
        embeddings: npt.NDArray[np.float32],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add embeddings, their string IDs, and metadata to the index.

        Supports batched and incremental indexing.

        Args:
            embeddings: A 2D numpy array of embeddings (np.float32).
            ids: A list of string IDs corresponding to the embeddings.
            metadata: An optional list of metadata dictionaries.

        Raises:
            ValueError: If input dimensions or lengths are invalid.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embeddings must be a 2D array with dimension {self.dimension}")

        num_vectors = embeddings.shape[0]
        if len(ids) != num_vectors:
            raise ValueError("Number of IDs must match the number of embeddings.")

        if metadata is None:
            metadata = [{} for _ in range(num_vectors)]
        elif len(metadata) != num_vectors:
            raise ValueError("Length of metadata must match the number of embeddings.")

        # Ensure embeddings are contiguous and float32
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)

        start_id = self.index.ntotal
        int_ids = list(range(start_id, start_id + num_vectors))

        # Add to Faiss index
        self.index.add(embeddings_f32)

        # Add to SQLite store
        self.store.add(int_ids, ids, metadata)

    def search(
        self, query_embeddings: npt.NDArray[np.float32], k: int = 10
    ) -> Tuple[
        npt.NDArray[np.float32],
        List[List[Optional[str]]],
        List[List[Optional[Dict[str, Any]]]],
    ]:
        """Search for the k nearest neighbors.

        Args:
            query_embeddings: 2D numpy array of query vectors.
            k: Number of nearest neighbors to retrieve.

        Returns:
            A tuple of (distances, string_ids, metadata).

        Raises:
            ValueError: If query embeddings dimensions are invalid.
        """
        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != self.dimension:
            raise ValueError(f"Query embeddings must be a 2D array with dimension {self.dimension}")

        if self.index.ntotal == 0:
            return (
                np.array([], dtype=np.float32).reshape(query_embeddings.shape[0], 0),
                [[] for _ in range(query_embeddings.shape[0])],
                [[] for _ in range(query_embeddings.shape[0])],
            )

        query_f32 = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        # Ensure k is not larger than index size
        actual_k = min(k, self.index.ntotal)

        distances, int_indices = self.index.search(query_f32, actual_k)

        all_string_ids: List[List[Optional[str]]] = []
        all_metadata: List[List[Optional[Dict[str, Any]]]] = []

        for indices in int_indices:
            # -1 is returned by Faiss if not enough results are found
            valid_indices = [int(idx) for idx in indices if idx != -1]
            if not valid_indices:
                all_string_ids.append([])
                all_metadata.append([])
                continue

            str_ids, meta = self.store.get_by_int_ids(valid_indices)

            # Reconstruct list to handle -1
            row_str_ids: List[Optional[str]] = []
            row_meta: List[Optional[Dict[str, Any]]] = []
            valid_idx_ptr = 0
            for idx in indices:
                if idx == -1:
                    row_str_ids.append(None)
                    row_meta.append(None)
                else:
                    row_str_ids.append(str_ids[valid_idx_ptr])
                    row_meta.append(meta[valid_idx_ptr])
                    valid_idx_ptr += 1

            all_string_ids.append(row_str_ids)
            all_metadata.append(row_meta)

        return distances, all_string_ids, all_metadata

    def save(self, index_path: str) -> None:
        """Save the Faiss index to disk.

        Args:
            index_path: The file path to save the index to.
        """
        faiss.write_index(self.index, index_path)

    def load(self, index_path: str) -> None:
        """Load the Faiss index from disk.

        Args:
            index_path: The file path to load the index from.

        Raises:
            FileNotFoundError: If the index file does not exist.
            ValueError: If the loaded index dimension does not match.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file {index_path} not found.")
        self.index = faiss.read_index(index_path)
        if self.index.d != self.dimension:
            raise ValueError(
                f"Loaded index dimension ({self.index.d}) does not match "
                f"expected dimension ({self.dimension})"
            )
