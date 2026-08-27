"""Faiss-based index implementation."""

import os
from typing import Any, cast

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
        self.index: faiss.IndexIDMap2 = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self.store = SQLiteStore(db_path)

    def add(
        self,
        embeddings: npt.NDArray[np.float32],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
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

        # The store treats string_id as unique and replaces on conflict. Evict the
        # superseded vectors first, otherwise they linger in Faiss with no store row
        # and silently consume result slots on every later search.
        superseded = self.store.get_int_ids_for_string_ids(list(ids))
        if superseded:
            self.remove_int_ids(superseded)

        int_ids = list(range(self.next_int_id(), self.next_int_id() + num_vectors))

        # Add to Faiss index with explicit IDs
        self.index.add_with_ids(embeddings_f32, np.array(int_ids, dtype=np.int64))

        # Add to SQLite store
        self.store.add(int_ids, ids, metadata)

    def next_int_id(self) -> int:
        """Return the next free integer ID.

        Derived from both the store and the Faiss index. Using the store alone
        restarts numbering at zero whenever the metadata database is lost or
        replaced, which collides with IDs still held by a loaded index file.

        Returns:
            An integer ID not currently used by either half of the index.
        """
        return max(self.store.get_max_int_id(), self.max_faiss_int_id()) + 1

    def max_faiss_int_id(self) -> int:
        """Return the largest ID present in the Faiss index, or -1 if empty.

        Returns:
            The maximum assigned ID.
        """
        if self.index.ntotal == 0:
            return -1
        ids = faiss.vector_to_array(self.index.id_map)
        return int(ids.max()) if ids.size else -1

    def desync_report(self) -> str | None:
        """Describe a mismatch between the vector index and the metadata store.

        Returns:
            A human-readable description, or None when the two agree.
        """
        vectors = int(self.index.ntotal)
        rows = self.store.count()
        if vectors == rows:
            return None
        return (
            f"index holds {vectors} vector(s) but the metadata store holds {rows} row(s); "
            f"the two files are out of sync"
        )

    def search(
        self, query_embeddings: npt.NDArray[np.float32], k: int = 10
    ) -> tuple[
        npt.NDArray[np.float32],
        list[list[str | None]],
        list[list[dict[str, Any] | None]],
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
            raise ValueError(
                f"Query embeddings must be a 2D array with dimension {self.dimension}"
            )

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

        all_string_ids: list[list[str | None]] = []
        all_metadata: list[list[dict[str, Any] | None]] = []

        for indices in int_indices:
            # -1 is returned by Faiss if not enough results are found
            valid_indices = [int(idx) for idx in indices if idx != -1]
            if not valid_indices:
                all_string_ids.append([])
                all_metadata.append([])
                continue

            str_ids, meta = self.store.get_by_int_ids(valid_indices)

            # Reconstruct list to handle -1
            row_str_ids: list[str | None] = []
            row_meta: list[dict[str, Any] | None] = []
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

    def remove_int_ids(self, int_ids: list[int]) -> None:
        """Remove vectors by their integer IDs from the FAISS index and metadata store.

        Args:
            int_ids: List of integer IDs to remove.
        """
        if not int_ids:
            return
        ids_array = np.array(int_ids, dtype=np.int64)
        self.index.remove_ids(faiss.IDSelectorBatch(ids_array))
        self.store.delete_by_int_ids(int_ids)

    def save(self, index_path: str) -> None:
        """Save the Faiss index to disk.

        Written to a sibling temp file and renamed, so an interrupted save
        leaves the previous index intact rather than a truncated one.

        Args:
            index_path: The file path to save the index to.
        """
        parent = os.path.dirname(index_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temp_path = f"{index_path}.tmp"
        faiss.write_index(self.index, temp_path)
        os.replace(temp_path, index_path)

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

        loaded = faiss.read_index(index_path)
        if loaded.d != self.dimension:
            raise ValueError(
                f"Loaded index dimension ({loaded.d}) does not match "
                f"expected dimension ({self.dimension})"
            )
        # Vectors are addressed by explicit IDs that join to the metadata store.
        # An index saved without an ID map cannot support that, and casting it
        # anyway only defers the failure to a raw error deep inside Faiss.
        if not hasattr(loaded, "id_map"):
            raise ValueError(
                f"Index file {index_path} has no ID map (type {type(loaded).__name__}), so its "
                f"vectors cannot be matched to stored metadata. It was written by an "
                f"incompatible version of EchoVector; re-index this store to rebuild it."
            )
        self.index = cast("faiss.IndexIDMap2", loaded)
