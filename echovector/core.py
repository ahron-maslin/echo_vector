"""Core public API for EchoVector."""

import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
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
from echovector.indexing.manifest import FORMAT_VERSION, IndexManifest, IndexMismatchError
from echovector.search.filters import SearchFilter
from echovector.search.results import SearchResult, TimestampRange

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aif"}

#: How many extra candidates to pull from the index when filtering, since
#: filters are applied after retrieval and can discard most of a page.
FILTER_OVERFETCH = 5

__all__ = ["AUDIO_EXTENSIONS", "EchoVector", "IndexMismatchError"]


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
        self.audio_processor = FileAudioProcessor(target_sample_rate=sample_rate, mono=True)

        self._backend_spec = backend
        self._backend_kwargs = backend_kwargs
        self._embedder: EmbeddingBackend | None = None if isinstance(backend, str) else backend

        self._stored_manifest = IndexManifest.read(self.store_dir)

        # A manifest records the embedding width, so a store can be opened — and
        # `stats()` answered — without paying to construct the model.
        dimension = (
            self._stored_manifest.embedding_dim
            if self._stored_manifest is not None
            else self.embedder.embedding_dim
        )
        self.index_backend = FaissIndex(dimension=dimension, db_path=str(self.db_path))

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

        Returns:
            Number of new chunks added to the index.

        Raises:
            ValueError: If batch_size is not positive, or chunks exceed the
                backend's embedding window.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        files = self._resolve_audio_files(
            targets,
            self.recursive if recursive is None else recursive,
        )
        if not files:
            return 0

        embedder = self.embedder
        self._assert_chunk_fits_backend(embedder)
        self._warn_on_chunking_drift(embedder)

        if force:
            for file_path in files:
                int_ids = self.index_backend.store.delete_by_filepath(str(file_path))
                if int_ids:
                    self.index_backend.remove_int_ids(int_ids)
        else:
            files = [f for f in files if not self.index_backend.store.has_filepath(str(f))]
            if not files:
                return 0

        indexed_chunks = 0
        with self._chunk_batcher(embedder, batch_size) as batcher:
            for file_number, file_path in enumerate(files, start=1):
                audio = self.audio_processor.load_audio(str(file_path))
                # Metadata describes the file, not the chunk, so read it once per
                # file rather than reopening the file for every chunk.
                file_metadata = extract_metadata(str(file_path))

                for chunk_number, start_s, end_s, chunk_audio in self._iter_chunks(audio):
                    batcher.add(
                        chunk_audio,
                        f"{file_path}#{start_s:.3f}-{end_s:.3f}",
                        self._metadata_for_chunk(
                            file_path, file_metadata, chunk_number, start_s, end_s
                        ),
                    )
                    indexed_chunks += 1

                if on_file_indexed is not None:
                    on_file_indexed(file_number, len(files), file_path)

        self._current_manifest(embedder).write(self.store_dir)
        self._stored_manifest = IndexManifest.read(self.store_dir)
        self.index_backend.save(str(self.index_path))
        return indexed_chunks

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilter | None = None,
    ) -> list[SearchResult]:
        """Search indexed audio with a text query.

        Args:
            query: Natural-language description of the audio to find.
            top_k: Maximum number of results to return.
            filters: Optional post-retrieval filters on score, filepath, or metadata.

        Returns:
            Matching chunks ordered by descending similarity.

        Raises:
            ValueError: If top_k is negative.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if top_k == 0:
            return []

        # Filters run after retrieval, so ask the index for extra candidates to
        # avoid returning fewer than top_k results that would have qualified.
        fetch_k = top_k * FILTER_OVERFETCH if filters is not None else top_k

        query_embedding = self._normalize_rows(self.embedder.embed_text([query]))
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
