"""Core public API for EchoVector."""

from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import soundfile as sf

from echovector.audio.metadata import extract_metadata
from echovector.audio.processor import AudioProcessor as FileAudioProcessor
from echovector.embeddings.base import EmbeddingBackend
from echovector.embeddings.factory import get_embedding_model
from echovector.indexing.faiss_index import FaissIndex
from echovector.search.results import SearchResult, TimestampRange

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aif"}


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
        """Initialize EchoVector."""
        if chunk_seconds <= 0.0:
            raise ValueError("chunk_seconds must be positive.")
        if overlap_seconds < 0.0:
            raise ValueError("overlap_seconds cannot be negative.")
        if overlap_seconds >= chunk_seconds:
            raise ValueError("overlap_seconds must be smaller than chunk_seconds.")

        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.faiss"
        self.db_path = self.store_dir / "metadata.sqlite"
        self.recursive = recursive
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.sample_rate = sample_rate
        self.audio_processor = FileAudioProcessor(target_sample_rate=sample_rate, mono=True)

        self.embedder = (
            get_embedding_model(backend, **backend_kwargs) if isinstance(backend, str) else backend
        )
        self.index_backend = FaissIndex(
            dimension=self.embedder.embedding_dim,
            db_path=str(self.db_path),
        )

        if self.index_path.exists():
            self.index_backend.load(str(self.index_path))

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
        """
        files = self._resolve_audio_files(
            targets,
            self.recursive if recursive is None else recursive,
        )
        if not files:
            return 0

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
        with TemporaryDirectory(prefix="echovector-chunks-") as temp_dir:
            chunk_paths: list[Path] = []
            chunk_ids: list[str] = []
            chunk_metadata: list[dict[str, Any]] = []

            for file_number, file_path in enumerate(files, start=1):
                audio = self.audio_processor.load_audio(str(file_path))
                for (
                    chunk_number,
                    start_seconds,
                    end_seconds,
                    chunk_audio,
                ) in self._iter_chunks(audio):
                    chunk_path = Path(temp_dir) / f"chunk-{indexed_chunks:08d}.wav"
                    sf.write(chunk_path, chunk_audio, self.sample_rate)
                    chunk_paths.append(chunk_path)
                    chunk_ids.append(f"{file_path}#{start_seconds:.3f}-{end_seconds:.3f}")
                    chunk_metadata.append(
                        self._metadata_for_chunk(
                            file_path,
                            chunk_number,
                            start_seconds,
                            end_seconds,
                        )
                    )
                    indexed_chunks += 1

                    if len(chunk_paths) >= batch_size:
                        self._add_chunk_batch(chunk_paths, chunk_ids, chunk_metadata)
                        chunk_paths = []
                        chunk_ids = []
                        chunk_metadata = []

                if on_file_indexed is not None:
                    on_file_indexed(file_number, len(files), file_path)

            if chunk_paths:
                self._add_chunk_batch(chunk_paths, chunk_ids, chunk_metadata)

        self.index_backend.save(str(self.index_path))
        return indexed_chunks

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search indexed audio with a text query."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if top_k == 0:
            return []

        query_embedding = self._normalize_rows(self.embedder.embed_text([query]))
        distances, ids, metadata_rows = self.index_backend.search(query_embedding, k=top_k)

        results: list[SearchResult] = []
        for offset, string_id in enumerate(ids[0] if ids else []):
            if string_id is None:
                continue
            metadata = metadata_rows[0][offset] if metadata_rows and metadata_rows[0] else {}
            metadata = metadata or {}
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

        return results

    def stats(self) -> dict[str, Any]:
        """Return basic index statistics."""
        return {
            "store_dir": str(self.store_dir),
            "index_path": str(self.index_path),
            "metadata_path": str(self.db_path),
            "embedding_dim": self.embedder.embedding_dim,
            "chunks": int(self.index_backend.index.ntotal),
            "vectors": int(self.index_backend.index.ntotal),
            "chunk_seconds": self.chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
        }

    def reset(self) -> None:
        """Clear the current on-disk and in-memory index."""
        if self.index_path.exists():
            self.index_path.unlink()
        if self.db_path.exists():
            self.index_backend.store.close()
            self.db_path.unlink()
        self.index_backend = FaissIndex(
            dimension=self.embedder.embedding_dim,
            db_path=str(self.db_path),
        )

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
                    candidate
                    for candidate in iterator
                    if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS
                )
            elif path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(path)
            elif not path.exists():
                raise FileNotFoundError(f"Audio path not found: {path}")

        return sorted(dict.fromkeys(files))

    def _metadata_for_chunk(
        self,
        path: Path,
        chunk_number: int,
        start_seconds: float,
        end_seconds: float,
    ) -> dict[str, Any]:
        metadata = extract_metadata(str(path))
        return {
            "filepath": str(path),
            "filename": path.name,
            "chunk_id": chunk_number,
            "start": start_seconds,
            "end": end_seconds,
            "chunk_duration": end_seconds - start_seconds,
            "duration": metadata.duration,
            "sample_rate": metadata.sample_rate,
            "channels": metadata.channels,
            "format": metadata.format,
            "file_size": metadata.file_size,
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

    def _add_chunk_batch(
        self,
        chunk_paths: Sequence[Path],
        chunk_ids: Sequence[str],
        chunk_metadata: Sequence[dict[str, Any]],
    ) -> None:
        embeddings = self.embedder.embed_audio([str(path) for path in chunk_paths])
        self.index_backend.add(
            self._normalize_rows(embeddings),
            list(chunk_ids),
            list(chunk_metadata),
        )

    def _normalize_rows(
        self,
        embeddings: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        embeddings_f32 = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return cast("npt.NDArray[np.float32]", embeddings_f32 / norms)
