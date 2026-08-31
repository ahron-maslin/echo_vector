"""Faiss-based index implementation."""

import os
from typing import Any, cast

import faiss
import numpy as np
import numpy.typing as npt

from .base import BaseIndex
from .store import SQLiteStore

#: Index structures `FaissIndex` knows how to build. Every one of them is
#: wrapped in an ``IndexIDMap2`` so vectors keep the explicit integer IDs that
#: join them to the metadata store.
INDEX_TYPES: tuple[str, ...] = ("flat", "ivf", "hnsw")


class FaissIndex(BaseIndex):
    """Faiss inner-product index whose vectors carry explicit integer IDs.

    Three structures are selectable through ``index_type``:

    - ``"flat"``: exhaustive ``IndexFlatIP``. Exact, but O(n) per query with
      every vector resident in RAM. The default, and the right choice for a
      personal library.
    - ``"ivf"``: ``IndexIVFFlat``. Approximate and much faster on large
      archives, but it must be trained before it can accept any vector.
    - ``"hnsw"``: ``IndexHNSWFlat``. Approximate, needs no training, and gives
      the best recall per unit of query time, but Faiss cannot delete from it
      in place, so removals rebuild the graph (see :meth:`remove_int_ids`).

    Attributes:
        dimension: Width of every vector in the index.
        index_type: Structure currently in use, one of :data:`INDEX_TYPES`.
        index: The live ``IndexIDMap2``.
        store: Metadata store holding one row per vector.
    """

    def __init__(
        self,
        dimension: int,
        db_path: str = ":memory:",
        *,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 24,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        """Initialize the Faiss index and the metadata store.

        Args:
            dimension: Dimensionality of the embeddings.
            db_path: Path to the SQLite store database.
            index_type: One of :data:`INDEX_TYPES`.
            nlist: IVF only. Number of Voronoi cells. Also the minimum number of
                vectors the first batch must carry, since training cannot
                produce more centroids than it has points.
            nprobe: IVF only. Cells visited per query. Higher is slower and more
                accurate; the default probes roughly a quarter of the index.
            hnsw_m: HNSW only. Neighbours retained per node.
            ef_construction: HNSW only. Search breadth while building the graph.
            ef_search: HNSW only. Search breadth per query.

        Raises:
            ValueError: If `index_type` is not a supported structure.
        """
        if index_type not in INDEX_TYPES:
            raise ValueError(
                f"Unknown index_type {index_type!r}; expected one of {', '.join(INDEX_TYPES)}."
            )

        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.index: faiss.IndexIDMap2 = self._build_index()
        self.store = SQLiteStore(db_path)

    # ------------------------------------------------------------------
    # Index construction and tuning
    # ------------------------------------------------------------------

    def _build_index(self) -> "faiss.IndexIDMap2":
        """Create an empty index of the configured type.

        Returns:
            An empty ``IndexIDMap2`` wrapping the configured structure.
        """
        base: faiss.Index
        if self.index_type == "ivf":
            base = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(self.dimension),
                self.dimension,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
        elif self.index_type == "hnsw":
            base = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efConstruction = self.ef_construction
        else:
            base = faiss.IndexFlatIP(self.dimension)

        wrapped = faiss.IndexIDMap2(base)
        self._apply_search_params(wrapped)
        return wrapped

    def _apply_search_params(self, index: "faiss.IndexIDMap2") -> None:
        """Push this session's query-time knobs onto an index.

        Faiss persists `nprobe` and `efSearch` inside the index file, so a
        loaded index would otherwise silently keep whatever the writer used.

        Args:
            index: The index to configure in place.
        """
        sub = faiss.downcast_index(index.index)
        if isinstance(sub, faiss.IndexIVF):
            sub.nprobe = self.nprobe
        elif isinstance(sub, faiss.IndexHNSW):
            sub.hnsw.efSearch = self.ef_search

    def _sub_index(self) -> "faiss.Index":
        """Return the concrete structure wrapped by the live ID map.

        Returns:
            The downcast sub-index.
        """
        return faiss.downcast_index(self.index.index)

    def _adopt_structure_of(self, index: "faiss.IndexIDMap2") -> None:
        """Describe an existing index with this instance's own settings.

        Called after :meth:`load`, so that a later rebuild reproduces the index
        that is actually on disk rather than the one this session asked for.

        Args:
            index: The index just read from disk.
        """
        sub = faiss.downcast_index(index.index)
        if isinstance(sub, faiss.IndexIVF):
            self.index_type = "ivf"
            self.nlist = int(sub.nlist)
        elif isinstance(sub, faiss.IndexHNSW):
            self.index_type = "hnsw"
            # Level 0 of an HNSW graph holds 2*M links per node.
            self.hnsw_m = int(cast("Any", sub.hnsw).nb_neighbors(0)) // 2 or self.hnsw_m
            self.ef_construction = int(sub.hnsw.efConstruction)
        else:
            self.index_type = "flat"

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(
        self,
        embeddings: npt.NDArray[np.float32],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add embeddings, their string IDs, and metadata to the index.

        Supports batched and incremental indexing.

        An IVF index is trained on the first batch it receives. If that batch
        carries fewer than `nlist` vectors there are not enough points to place
        the centroids, and this raises before anything is written, leaving the
        index and store untouched.

        Args:
            embeddings: A 2D numpy array of embeddings (np.float32).
            ids: A list of string IDs corresponding to the embeddings.
            metadata: An optional list of metadata dictionaries.

        Raises:
            ValueError: If input dimensions or lengths are invalid, or if an
                untrained IVF index was given too few vectors to train on.
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

        if num_vectors == 0:
            return

        # Ensure embeddings are contiguous and float32
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Training can fail on a too-small first batch, so it runs before any
        # eviction: a rejected add must not have already dropped live vectors.
        self._ensure_trained(embeddings_f32)

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

    def _ensure_trained(self, embeddings: npt.NDArray[np.float32]) -> None:
        """Train the index on its first batch if the structure requires it.

        Args:
            embeddings: The batch about to be added.

        Raises:
            ValueError: If the batch is too small to train on.
        """
        if self.index.is_trained:
            return

        required = int(cast("Any", self._sub_index()).nlist)
        available = int(embeddings.shape[0])
        if available < required:
            raise ValueError(
                f"An IVF index must be trained on at least nlist={required} vectors before it "
                f"can hold any, but the first batch carries {available}. Index more audio in a "
                f"single call, lower nlist, or use index_type='flat' for a small library."
            )
        self.index.train(embeddings)

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

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def search(
        self, query_embeddings: npt.NDArray[np.float32], k: int = 10
    ) -> tuple[
        npt.NDArray[np.float32],
        list[list[str | None]],
        list[list[dict[str, Any] | None]],
    ]:
        """Search for the k nearest neighbors.

        Approximate index types may return fewer than k hits, or hits that a
        flat index would have ranked lower; those slots come back as None
        rather than being dropped, so the caller sees the shape it asked for.

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

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def supports_native_removal(self) -> bool:
        """Report whether Faiss can delete from the live structure in place.

        Returns:
            False for HNSW, whose graph Faiss cannot edit; True otherwise.
        """
        return not isinstance(self._sub_index(), faiss.IndexHNSW)

    def remove_int_ids(self, int_ids: list[int]) -> None:
        """Remove vectors by their integer IDs from the FAISS index and metadata store.

        Faiss refuses `remove_ids` on an HNSW graph. Leaving those vectors in
        place would recreate the orphan bug this index guards against, so for
        HNSW the surviving vectors are reconstructed into a fresh graph
        instead. That is O(n) in the size of the index, which makes deletes and
        forced re-indexing markedly more expensive on an HNSW store.

        Args:
            int_ids: List of integer IDs to remove.
        """
        if not int_ids:
            return
        if self.supports_native_removal():
            self.index.remove_ids(faiss.IDSelectorBatch(np.array(int_ids, dtype=np.int64)))
        else:
            self._remove_by_rebuild(int_ids)
        self.store.delete_by_int_ids(int_ids)

    def _remove_by_rebuild(self, int_ids: list[int]) -> None:
        """Rebuild the index from the vectors that are not being removed.

        Args:
            int_ids: Integer IDs to drop.
        """
        total = int(self.index.ntotal)
        if total == 0:
            return

        # id_map is ordered by insertion, which is exactly the ordering
        # reconstruct_n uses, so the two arrays line up row for row.
        existing_ids = faiss.vector_to_array(self.index.id_map)
        vectors = cast("Any", self._sub_index()).reconstruct_n(0, total)

        dropped = set(int_ids)
        keep = np.array([int(i) not in dropped for i in existing_ids], dtype=bool)

        rebuilt = self._build_index()
        if bool(keep.any()):
            rebuilt.add_with_ids(
                np.ascontiguousarray(vectors[keep], dtype=np.float32),
                np.ascontiguousarray(existing_ids[keep], dtype=np.int64),
            )
        self.index = rebuilt

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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

        The structure on disk wins over the one this instance was constructed
        with: `index_type` and the structural knobs are re-read from the file,
        while query-time knobs (`nprobe`, `ef_search`) stay as configured here.

        Args:
            index_path: The file path to load the index from.

        Raises:
            FileNotFoundError: If the index file does not exist.
            ValueError: If the loaded index dimension does not match, or if it
                carries no ID map.
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
        index = cast("faiss.IndexIDMap2", loaded)
        self._adopt_structure_of(index)
        self._apply_search_params(index)
        self.index = index
