"""Core public API for EchoVector."""

import warnings
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from itertools import islice
from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import soundfile as sf

from echovector.audio.metadata import AudioMetadata, extract_metadata
from echovector.audio.processor import AudioProcessor as FileAudioProcessor
from echovector.embeddings.base import (
    EmbeddingBackend,
    WaveformEmbeddingBackend,
    backend_fingerprint,
    max_audio_seconds,
)
from echovector.embeddings.factory import get_embedding_model
from echovector.indexing.faiss_index import FaissIndex
from echovector.indexing.manifest import (
    DEFAULT_INDEX_TYPE,
    FORMAT_VERSION,
    IndexManifest,
    IndexMismatchError,
)
from echovector.search.filters import SearchFilter
from echovector.search.results import SearchResult, TimestampRange
from echovector.search.spans import limit_per_file, merge_adjacent_spans

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aif"}

#: How many extra candidates to pull from the index when filtering, since
#: filters are applied after retrieval and can discard most of a page.
FILTER_OVERFETCH = 5

#: A neighbouring chunk joins a span only if it scores at least this fraction of
#: the file's best hit. Adjacency alone would let a strong hit absorb the
#: unrelated audio next to it and report a span the user did not ask for.
SPAN_MERGE_RELATIVE_SCORE = 0.9

#: Slack when comparing a file's stored mtime with the one on disk. Filesystems
#: and archive tools round timestamps differently, so an exact match would
#: report unchanged files as stale.
MTIME_TOLERANCE_SECONDS = 1e-6

__all__ = ["AUDIO_EXTENSIONS", "EchoVector", "IndexMismatchError"]

#: One file's decoded audio, its metadata, and its modification time.
_DecodedFile = tuple[npt.NDArray[np.float32], AudioMetadata, float]


class EchoVector:
    """High-level interface for indexing and searching audio files."""

    def __init__(
        self,
        store_dir: str | Path = ".echovector",
        backend: str | EmbeddingBackend = "clap",
        recursive: bool = True,
        chunk_seconds: float = 10.0,
        overlap_seconds: float = 1.0,
        sample_rate: int = 48_000,
        index_type: str = DEFAULT_INDEX_TYPE,
        index_params: dict[str, Any] | None = None,
        **backend_kwargs: Any,
    ) -> None:
        """Initialize EchoVector.

        Args:
            store_dir: Directory holding the vector index, metadata store, and manifest.
            backend: Embedding backend name, or an already-constructed backend.
            recursive: Whether directory targets are searched recursively.
            chunk_seconds: Length of each embedded audio chunk.
            overlap_seconds: Overlap between adjacent chunks.
            sample_rate: Sample rate audio is resampled to before chunking.
            index_type: Vector index structure: "flat" (exact, the default),
                or "ivf"/"hnsw" for approximate search over a large archive.
            index_params: Tuning knobs for the chosen structure, e.g.
                {"nlist": 256} for IVF or {"ef_search": 128} for HNSW.
            **backend_kwargs: Forwarded to the backend constructor.

        Raises:
            ValueError: If the chunking parameters are not usable.
            IndexMismatchError: If an existing index was built incompatibly.
        """
        if chunk_seconds <= 0.0:
            raise ValueError("chunk_seconds must be positive.")
        if overlap_seconds < 0.0:
            raise ValueError("overlap_seconds cannot be negative.")
        if overlap_seconds >= chunk_seconds:
            raise ValueError("overlap_seconds must be smaller than chunk_seconds.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.faiss"
        self.db_path = self.store_dir / "metadata.sqlite"
        self.recursive = recursive
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.sample_rate = sample_rate
        self.index_type = index_type
        self.index_params = dict(index_params or {})
        self.audio_processor = FileAudioProcessor(target_sample_rate=sample_rate, mono=True)

        self._backend_spec = backend
        self._backend_kwargs = backend_kwargs
        self._embedder: EmbeddingBackend | None = None if isinstance(backend, str) else backend

        self._stored_manifest = IndexManifest.read(self.store_dir)

        # An existing store's structure is whatever is on disk. Adopt it rather
        # than pretending the request took effect, but say so: switching
        # structures changes recall and speed and needs a re-index.
        if self._stored_manifest is not None:
            if self._stored_manifest.index_type != index_type:
                warnings.warn(
                    f"Store at {self.store_dir} was built as "
                    f"'{self._stored_manifest.index_type}', not '{index_type}'. Using the "
                    f"structure on disk; re-index with reset=True to change it.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.index_type = self._stored_manifest.index_type

        # A manifest records the embedding width, so a store can be opened — and
        # `stats()` answered — without paying to construct the model.
        dimension = (
            self._stored_manifest.embedding_dim
            if self._stored_manifest is not None
            else self.embedder.embedding_dim
        )
        self.index_backend = FaissIndex(
            dimension=dimension,
            db_path=str(self.db_path),
            index_type=self.index_type,
            **self.index_params,
        )

        if self.index_path.exists():
            self.index_backend.load(str(self.index_path))

        desync = self.index_backend.desync_report()
        if desync is not None:
            warnings.warn(
                f"EchoVector store at {self.store_dir} is inconsistent: {desync}. "
                f"Searches may return incomplete results; re-index with reset=True to repair.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------

    @property
    def embedder(self) -> EmbeddingBackend:
        """Return the embedding backend, constructing it on first use.

        Deferring construction keeps metadata-only operations (`stats`) free of
        a multi-hundred-megabyte model load.

        Returns:
            The embedding backend for this store.

        Raises:
            IndexMismatchError: If the backend cannot search the existing index.
        """
        if self._embedder is None:
            self._embedder = get_embedding_model(
                cast("str", self._backend_spec), **self._backend_kwargs
            )
        self._assert_backend_matches_index(self._embedder)
        return self._embedder

    def _current_manifest(self, embedder: EmbeddingBackend) -> IndexManifest:
        return IndexManifest(
            format_version=FORMAT_VERSION,
            backend=backend_fingerprint(embedder),
            embedding_dim=embedder.embedding_dim,
            chunk_seconds=self.chunk_seconds,
            overlap_seconds=self.overlap_seconds,
            sample_rate=self.sample_rate,
            index_type=self.index_type,
        )

    def _assert_backend_matches_index(self, embedder: EmbeddingBackend) -> None:
        if self._stored_manifest is None:
            return
        self._stored_manifest.assert_compatible_with(
            self._current_manifest(embedder), self.store_dir
        )

    def _warn_on_chunking_drift(self, embedder: EmbeddingBackend) -> None:
        """Warn when new chunks will not line up with the ones already stored.

        Only relevant when adding audio, so this is checked at index time rather
        than on open — `stats` and `search` never re-chunk anything.
        """
        stored = self._stored_manifest
        if stored is None:
            return
        differences = stored.chunking_differences(self._current_manifest(embedder))
        if differences:
            warnings.warn(
                f"Store at {self.store_dir} was indexed with different chunking settings: "
                f"{'; '.join(differences)}. Existing chunks are unaffected, but newly indexed "
                f"audio will be segmented differently.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _assert_chunk_fits_backend(self, embedder: EmbeddingBackend) -> None:
        limit = max_audio_seconds(embedder)
        if limit is not None and self.chunk_seconds > limit:
            raise ValueError(
                f"chunk_seconds={self.chunk_seconds} exceeds the {limit}s window this backend "
                f"can embed. Longer chunks are cropped before embedding, so the vector would "
                f"not describe the timestamp range stored with it. Use chunk_seconds <= {limit}."
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        targets: str | Path | Sequence[str | Path],
        recursive: bool | None = None,
        batch_size: int = 16,
        force: bool = False,
        on_file_indexed: Callable[[int, int, Path], None] | None = None,
        checkpoint_every_chunks: int = 2_000,
        decode_workers: int = 4,
    ) -> int:
        """Index audio chunks from paths or directories.

        Args:
            targets: One or more file paths or directories to index.
            recursive: Override the instance-level recursive setting.
            batch_size: Number of chunks to embed per batch.
            force: If True, remove and re-index files that are already stored.
                   If False (default), already-indexed files are skipped.
            on_file_indexed: Optional callback invoked after each file is
                processed, with (files_done, total_files, file_path). Useful
                for reporting progress.
            checkpoint_every_chunks: Save the vector index to disk after roughly
                this many chunks, so an interrupted run keeps its progress.
                Set to 0 to save only at the end.
            decode_workers: Threads used to read audio ahead of the embedder, so
                decoding overlaps with embedding instead of alternating. Set to
                1 to decode inline.

        Returns:
            Number of new chunks added to the index.

        Raises:
            ValueError: If batch_size is not positive, or chunks exceed the
                backend's embedding window.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if decode_workers < 1:
            raise ValueError("decode_workers must be at least 1.")

        files = self._resolve_audio_files(
            targets,
            self.recursive if recursive is None else recursive,
        )
        if not files:
            return 0

        embedder = self.embedder
        self._assert_chunk_fits_backend(embedder)
        self._warn_on_chunking_drift(embedder)

        if not force:
            # An unchanged file is skipped, but one edited or replaced since it
            # was indexed must be re-read: matching on path alone would leave
            # the index describing audio that no longer exists.
            files = [f for f in files if self._has_changed(f)]
            if not files:
                return 0

        for file_path in files:
            int_ids = self.index_backend.store.delete_by_filepath(str(file_path))
            if int_ids:
                self.index_backend.remove_int_ids(int_ids)

        # Written before the loop so an interrupted run still leaves a store that
        # describes itself, rather than vectors with no recorded provenance.
        self._current_manifest(embedder).write(self.store_dir)
        self._stored_manifest = IndexManifest.read(self.store_dir)

        indexed_chunks = 0
        since_checkpoint = 0
        with self._chunk_batcher(embedder, batch_size) as batcher:
            for file_number, (file_path, decoded) in enumerate(
                self._decoded_files(files, decode_workers), start=1
            ):
                audio, file_metadata, file_mtime = decoded

                for chunk_number, start_s, end_s, chunk_audio in self._iter_chunks(audio):
                    batcher.add(
                        chunk_audio,
                        f"{file_path}#{start_s:.3f}-{end_s:.3f}",
                        self._metadata_for_chunk(
                            file_path, file_metadata, chunk_number, start_s, end_s, file_mtime
                        ),
                    )
                    indexed_chunks += 1
                    since_checkpoint += 1

                if on_file_indexed is not None:
                    on_file_indexed(file_number, len(files), file_path)

                if checkpoint_every_chunks > 0 and since_checkpoint >= checkpoint_every_chunks:
                    self._checkpoint(batcher)
                    since_checkpoint = 0

        self.index_backend.save(str(self.index_path))
        return indexed_chunks

    def _decode_one(self, file_path: Path) -> _DecodedFile:
        """Read one file's audio, metadata, and signature.

        Args:
            file_path: Canonical path to the audio file.

        Returns:
            The decoded waveform, the file's metadata, and its modification time.
        """
        audio = self.audio_processor.load_audio(str(file_path))
        # Metadata describes the file, not the chunk, so read it once per file
        # rather than reopening the file for every chunk.
        return audio, extract_metadata(str(file_path)), file_path.stat().st_mtime

    def _decoded_files(
        self,
        files: Sequence[Path],
        decode_workers: int,
    ) -> Iterator[tuple[Path, _DecodedFile]]:
        """Yield each file with its decoded audio, in order.

        Reading the next files while the current batch is embedded keeps the
        model fed instead of alternating between the two. Order is preserved so
        progress callbacks and chunk ids stay deterministic, and only a small
        window is held in memory at once.

        How much this wins depends entirely on how much of each side releases
        the GIL. Measured on 40 clips: ~1.5x with compressed audio and a backend
        that releases it during inference (torch, so CLAP), ~1.3x for compressed
        audio alone, and nothing at all — slightly worse past two workers — for
        uncompressed WAV through the GIL-bound `local` backend. It is a modest
        win in the common case, not a transformative one.

        Args:
            files: Canonical audio paths, in the order they should be indexed.
            decode_workers: Threads used to read ahead. 1 decodes inline.

        Yields:
            Pairs of (path, decoded), in the same order as `files`.
        """
        if decode_workers <= 1 or len(files) == 1:
            for file_path in files:
                yield file_path, self._decode_one(file_path)
            return

        # Bounded so a large corpus of long recordings cannot be decoded into
        # memory faster than the embedder consumes it.
        window = max(2, decode_workers * 2)
        with ThreadPoolExecutor(
            max_workers=decode_workers, thread_name_prefix="echovector-decode"
        ) as pool:
            pending: deque[tuple[Path, Future[_DecodedFile]]] = deque()
            upcoming = iter(files)

            for file_path in islice(upcoming, window):
                pending.append((file_path, pool.submit(self._decode_one, file_path)))

            while pending:
                done_path, future = pending.popleft()
                for file_path in islice(upcoming, 1):
                    pending.append((file_path, pool.submit(self._decode_one, file_path)))
                yield done_path, future.result()

    def _checkpoint(self, batcher: "_ChunkBatcher") -> None:
        """Persist progress so far.

        The store commits per batch but the vector index only reaches disk on
        save, so an interrupted long run would otherwise leave metadata rows
        with no matching vectors. Flushing before saving is what keeps the two
        halves consistent at the checkpoint.
        """
        batcher.flush()
        self.index_backend.save(str(self.index_path))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilter | None = None,
        merge_spans: bool = True,
        max_gap_seconds: float = 0.0,
        max_per_file: int | None = None,
    ) -> list[SearchResult]:
        """Search indexed audio with a text query.

        Args:
            query: Natural-language description of the audio to find.
            top_k: Maximum number of results to return.
            filters: Optional post-retrieval filters on score, filepath, or metadata.
            merge_spans: Fold consecutive hits from one file into a single
                moment. A 40-second passage is otherwise returned as four or
                five near-identical chunks that crowd out every other file.
                Set False for the raw chunk-level results.
            max_gap_seconds: When merging, the largest silence between two hits
                that still counts as one moment.
            max_per_file: Optionally cap how many results any one file may
                contribute, so a single dominant recording cannot fill the page.

        Returns:
            Matching moments ordered by descending similarity.

        Raises:
            ValueError: If top_k is negative.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if top_k == 0:
            return []

        query_embedding = self.embedder.embed_text([query])
        return self._search_embedding(
            query_embedding,
            top_k,
            filters,
            merge_spans=merge_spans,
            max_gap_seconds=max_gap_seconds,
            max_per_file=max_per_file,
        )

    def search_by_audio(
        self,
        audio: str | Path | npt.NDArray[np.float32],
        top_k: int = 5,
        sample_rate: int | None = None,
        filters: SearchFilter | None = None,
        exclude_source: bool = True,
    ) -> list[SearchResult]:
        """Find indexed audio that sounds like the given clip.

        Unlike text search this never touches the text tower, so it works with
        audio-only backends (wav2vec2, HuBERT, AST) that have no aligned text
        embedding at all.

        Args:
            audio: Path to an audio file, or a mono float32 waveform.
            top_k: Maximum number of results to return.
            sample_rate: Rate of `audio` when passing a waveform. Defaults to
                the engine's own sample rate.
            filters: Optional post-retrieval filters.
            exclude_source: Drop hits from the query file itself, which would
                otherwise take every top slot when the query is already indexed.

        Returns:
            Matching chunks ordered by descending similarity.

        Raises:
            ValueError: If top_k is negative, or a waveform is not 1-D.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if top_k == 0:
            return []

        source_path: Path | None = None
        if isinstance(audio, (str, Path)):
            source_path = self._canonical(audio)
            waveform = self.audio_processor.load_audio(str(source_path))
            rate = self.sample_rate
        else:
            waveform = np.asarray(audio, dtype=np.float32)
            if waveform.ndim != 1:
                raise ValueError("Query waveform must be mono (1-D).")
            rate = sample_rate if sample_rate is not None else self.sample_rate

        embedder = self.embedder
        if isinstance(embedder, WaveformEmbeddingBackend):
            query_embedding = embedder.embed_audio_arrays([waveform], rate)
        else:
            query_embedding = self._embed_waveform_via_temp_file(embedder, waveform, rate)

        # Excluding the source removes candidates after retrieval, so widen the
        # page to avoid coming back short.
        fetch_k = top_k * FILTER_OVERFETCH if (exclude_source and source_path) else top_k
        results = self._search_embedding(query_embedding, fetch_k, filters)

        if exclude_source and source_path is not None:
            source = str(source_path)
            results = [r for r in results if r.filepath != source]

        return results[:top_k]

    def _embed_waveform_via_temp_file(
        self,
        embedder: EmbeddingBackend,
        waveform: npt.NDArray[np.float32],
        rate: int,
    ) -> npt.NDArray[np.float32]:
        """Embed a waveform through a backend that only accepts file paths."""
        with TemporaryDirectory(prefix="echovector-query-") as temp_dir:
            path = Path(temp_dir) / "query.wav"
            sf.write(path, waveform, rate, subtype="FLOAT")
            return embedder.embed_audio([str(path)])

    def _search_embedding(
        self,
        embedding: npt.NDArray[np.float32],
        top_k: int,
        filters: SearchFilter | None,
        merge_spans: bool = False,
        max_gap_seconds: float = 0.0,
        max_per_file: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve and hydrate the nearest chunks to a single query vector."""
        # Filtering, merging, and per-file capping all remove candidates after
        # retrieval, so widen the page or the caller comes back short of top_k.
        reduces_results = filters is not None or merge_spans or max_per_file is not None
        fetch_k = top_k * FILTER_OVERFETCH if reduces_results else top_k

        query_embedding = self._normalize_rows(embedding)
        distances, ids, metadata_rows = self.index_backend.search(query_embedding, k=fetch_k)

        results: list[SearchResult] = []
        for offset, string_id in enumerate(ids[0] if ids else []):
            if string_id is None:
                continue
            row = metadata_rows[0][offset] if metadata_rows and metadata_rows[0] else None
            metadata = row or {}
            results.append(
                SearchResult(
                    filepath=str(metadata.get("filepath", string_id)),
                    timestamp_range=TimestampRange(
                        start=float(metadata.get("start", 0.0)),
                        end=float(metadata.get("end", metadata.get("duration", 0.0))),
                    ),
                    score=float(distances[0][offset]),
                    metadata=metadata,
                )
            )

        if filters is not None:
            results = filters.apply(results)
        if merge_spans:
            results = merge_adjacent_spans(
                results,
                max_gap_seconds=max_gap_seconds,
                min_relative_score=SPAN_MERGE_RELATIVE_SCORE,
            )
        if max_per_file is not None:
            results = limit_per_file(results, max_per_file)

        return results[:top_k]

    # ------------------------------------------------------------------
    # Introspection and maintenance
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return index statistics.

        Reads only the manifest and the stores, so it never loads a model.

        Returns:
            A mapping of statistic name to value.
        """
        manifest = self._stored_manifest
        return {
            "store_dir": str(self.store_dir),
            "index_path": str(self.index_path),
            "metadata_path": str(self.db_path),
            "backend": manifest.backend if manifest else "<not yet indexed>",
            "index_type": self.index_backend.index_type,
            "embedding_dim": self.index_backend.dimension,
            "files": self.index_backend.store.count_files(),
            "chunks": int(self.index_backend.index.ntotal),
            "chunk_seconds": manifest.chunk_seconds if manifest else self.chunk_seconds,
            "overlap_seconds": manifest.overlap_seconds if manifest else self.overlap_seconds,
            "sample_rate": manifest.sample_rate if manifest else self.sample_rate,
        }

    def reset(self) -> None:
        """Clear the current on-disk and in-memory index."""
        self.index_backend.store.close()
        for path in (self.index_path, self.db_path, IndexManifest.path_for(self.store_dir)):
            path.unlink(missing_ok=True)

        self._stored_manifest = None
        self.index_backend = FaissIndex(
            dimension=self.index_backend.dimension,
            db_path=str(self.db_path),
            index_type=self.index_type,
            **self.index_params,
        )

    def resolve_audio_files(
        self,
        targets: str | Path | Sequence[str | Path],
        recursive: bool | None = None,
    ) -> list[Path]:
        """Resolve targets (files and/or directories) to a sorted list of audio files.

        Paths are canonicalized, so the same file reached through a relative
        path, an absolute path, or a symlink resolves to one entry.

        Args:
            targets: One or more file paths or directories.
            recursive: Override the instance-level recursive setting.

        Returns:
            Sorted, de-duplicated list of audio file paths.
        """
        return self._resolve_audio_files(
            targets, self.recursive if recursive is None else recursive
        )

    def is_indexed(self, file_path: str | Path) -> bool:
        """Return True if any chunk from file_path is already stored.

        Args:
            file_path: Path to an audio file, in any equivalent spelling.
        """
        return self.index_backend.store.has_filepath(str(self._canonical(file_path)))

    def is_stale(self, file_path: str | Path) -> bool:
        """Return True if file_path has changed since it was indexed.

        Args:
            file_path: Path to an audio file, in any equivalent spelling.

        Returns:
            True when the file is unindexed, missing a recorded signature, or
            its size/mtime no longer match what was stored.
        """
        return self._has_changed(self._canonical(file_path))

    def prune(self) -> list[str]:
        """Drop chunks whose source file no longer exists on disk.

        Deleted recordings otherwise stay searchable forever, returning hits
        that point at nothing.

        Returns:
            The filepaths that were removed, sorted.
        """
        removed: list[str] = []
        for filepath in self.index_backend.store.list_filepaths():
            if Path(filepath).exists():
                continue
            int_ids = self.index_backend.store.delete_by_filepath(filepath)
            if int_ids:
                self.index_backend.remove_int_ids(int_ids)
            removed.append(filepath)

        if removed:
            self.index_backend.save(str(self.index_path))
        return sorted(removed)

    def stale_files(self) -> list[str]:
        """Return indexed files that have changed or vanished since indexing.

        Returns:
            Sorted filepaths needing a re-index.
        """
        return sorted(
            path
            for path in self.index_backend.store.list_filepaths()
            if self._has_changed(Path(path))
        )

    def _has_changed(self, file_path: Path) -> bool:
        """Return True when a file must be (re-)indexed."""
        stored = self.index_backend.store.get_file_signature(str(file_path))
        if stored is None:
            # Either never indexed, or indexed before signatures were recorded;
            # both mean the store cannot vouch for what it holds.
            return True
        try:
            current = file_path.stat()
        except OSError:
            # The file is gone. `prune()` removes it; re-reading is impossible.
            return False
        stored_mtime, stored_size = stored
        return stored_size != current.st_size or not isclose(
            stored_mtime, current.st_mtime, rel_tol=0.0, abs_tol=MTIME_TOLERANCE_SECONDS
        )

    def close(self) -> None:
        """Release the metadata store connection."""
        self.index_backend.store.close()

    def __enter__(self) -> "EchoVector":
        """Enter a context that closes the store on exit."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the store."""
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical(path: str | Path) -> Path:
        """Return a stable absolute path used as a file's identity in the store.

        Indexing the same audio through `./meetings`, `meetings`, and an
        absolute path must not produce three copies of its vectors.
        """
        expanded = Path(path).expanduser()
        try:
            return expanded.resolve()
        except OSError:  # pragma: no cover - resolve() on an unreadable mount
            return expanded.absolute()

    def _resolve_audio_files(
        self,
        targets: str | Path | Sequence[str | Path],
        recursive: bool,
    ) -> list[Path]:
        target_list = [targets] if isinstance(targets, (str, Path)) else targets
        files: list[Path] = []

        for target in target_list:
            path = Path(target).expanduser()
            if path.is_dir():
                iterator = path.rglob("*") if recursive else path.glob("*")
                files.extend(
                    self._canonical(candidate)
                    for candidate in iterator
                    if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS
                )
            elif path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(self._canonical(path))
            elif not path.exists():
                raise FileNotFoundError(f"Audio path not found: {path}")

        return sorted(dict.fromkeys(files))

    def _metadata_for_chunk(
        self,
        path: Path,
        file_metadata: AudioMetadata,
        chunk_number: int,
        start_seconds: float,
        end_seconds: float,
        file_mtime: float,
    ) -> dict[str, Any]:
        return {
            "filepath": str(path),
            "filename": path.name,
            "chunk_id": chunk_number,
            "start": start_seconds,
            "end": end_seconds,
            "chunk_duration": end_seconds - start_seconds,
            "duration": file_metadata.duration,
            "sample_rate": file_metadata.sample_rate,
            "channels": file_metadata.channels,
            "format": file_metadata.format,
            "file_size": file_metadata.file_size,
            # Paired with file_size as the signature change detection compares.
            "file_mtime": file_mtime,
        }

    def _iter_chunks(
        self,
        audio: npt.NDArray[np.float32],
    ) -> list[tuple[int, float, float, npt.NDArray[np.float32]]]:
        chunk_samples = round(self.chunk_seconds * self.sample_rate)
        overlap_samples = round(self.overlap_seconds * self.sample_rate)
        step_samples = chunk_samples - overlap_samples

        chunks: list[tuple[int, float, float, npt.NDArray[np.float32]]] = []
        if len(audio) == 0:
            return chunks

        start_sample = 0
        chunk_number = 0
        while start_sample < len(audio):
            end_sample = min(start_sample + chunk_samples, len(audio))
            chunk = audio[start_sample:end_sample]
            start_seconds = start_sample / self.sample_rate
            end_seconds = end_sample / self.sample_rate
            chunks.append((chunk_number, start_seconds, end_seconds, chunk))

            if end_sample == len(audio):
                break
            start_sample += step_samples
            chunk_number += 1

        return chunks

    @contextmanager
    def _chunk_batcher(
        self,
        embedder: EmbeddingBackend,
        batch_size: int,
    ) -> Iterator["_ChunkBatcher"]:
        """Yield a batcher that embeds chunks and flushes any partial batch on exit."""
        if isinstance(embedder, WaveformEmbeddingBackend):
            batcher = _ChunkBatcher(self, embedder, batch_size, temp_dir=None)
            yield batcher
            batcher.flush()
            return

        # Backends that only accept file paths still work, but each chunk has to
        # be written out and decoded again. Temp files are removed per batch so
        # scratch usage stays bounded by batch_size rather than corpus size.
        with TemporaryDirectory(prefix="echovector-chunks-") as temp_dir:
            batcher = _ChunkBatcher(self, embedder, batch_size, temp_dir=Path(temp_dir))
            yield batcher
            batcher.flush()

    def _normalize_rows(
        self,
        embeddings: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        embeddings_f32 = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return cast("npt.NDArray[np.float32]", embeddings_f32 / norms)


class _ChunkBatcher:
    """Accumulates chunks and embeds them a batch at a time."""

    def __init__(
        self,
        owner: EchoVector,
        embedder: EmbeddingBackend,
        batch_size: int,
        temp_dir: Path | None,
    ) -> None:
        self._owner = owner
        self._embedder = embedder
        self._batch_size = batch_size
        self._temp_dir = temp_dir
        self._waveforms: list[npt.NDArray[np.float32]] = []
        self._ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []

    def add(
        self,
        waveform: npt.NDArray[np.float32],
        chunk_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Queue one chunk, flushing when the batch is full."""
        self._waveforms.append(waveform)
        self._ids.append(chunk_id)
        self._metadata.append(metadata)
        if len(self._waveforms) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        """Embed and store everything queued so far."""
        if not self._waveforms:
            return

        embeddings = self._embed(self._waveforms)
        self._owner.index_backend.add(
            self._owner._normalize_rows(embeddings),
            list(self._ids),
            list(self._metadata),
        )
        self._waveforms = []
        self._ids = []
        self._metadata = []

    def _embed(self, waveforms: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        if self._temp_dir is None:
            backend = cast("WaveformEmbeddingBackend", self._embedder)
            return backend.embed_audio_arrays(waveforms, self._owner.sample_rate)

        paths: list[Path] = []
        try:
            for position, waveform in enumerate(waveforms):
                path = self._temp_dir / f"chunk-{position:05d}.wav"
                # soundfile defaults float input to PCM_16, which would quantize
                # every chunk to 16 bits on its way to the embedder.
                sf.write(path, waveform, self._owner.sample_rate, subtype="FLOAT")
                paths.append(path)
            return self._embedder.embed_audio([str(path) for path in paths])
        finally:
            for path in paths:
                path.unlink(missing_ok=True)
