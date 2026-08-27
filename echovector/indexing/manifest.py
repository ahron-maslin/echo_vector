"""Provenance record describing how an on-disk index was built.

Vectors on their own carry no record of the model that produced them. Without
one, an index built with CLAP can be queried with a different backend of the
same width and will return confident, meaningless scores. The manifest pins
that identity to the store directory so a mismatch fails loudly instead.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"

#: Bumped when the on-disk layout changes in a way older readers cannot handle.
FORMAT_VERSION = 1


class IndexMismatchError(ValueError):
    """Raised when an index was built with settings incompatible with the current ones."""


@dataclass(frozen=True)
class IndexManifest:
    """Settings that produced the vectors in a store directory.

    Attributes:
        format_version: On-disk layout version.
        backend: Fingerprint of the embedding backend, including model weights.
        embedding_dim: Width of every vector in the index.
        chunk_seconds: Chunk length used when the audio was sliced.
        overlap_seconds: Overlap between adjacent chunks.
        sample_rate: Sample rate audio was resampled to before embedding.
    """

    format_version: int
    backend: str
    embedding_dim: int
    chunk_seconds: float
    overlap_seconds: float
    sample_rate: int

    @classmethod
    def path_for(cls, store_dir: str | Path) -> Path:
        """Return the manifest path inside a store directory.

        Args:
            store_dir: The index store directory.

        Returns:
            Path to the manifest file, which may not exist.
        """
        return Path(store_dir) / MANIFEST_FILENAME

    @classmethod
    def read(cls, store_dir: str | Path) -> "IndexManifest | None":
        """Load the manifest for a store directory.

        Args:
            store_dir: The index store directory.

        Returns:
            The manifest, or None if the store predates manifests or has none.

        Raises:
            IndexMismatchError: If the manifest is unreadable or was written by
                a newer, incompatible version of EchoVector.
        """
        path = cls.path_for(store_dir)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexMismatchError(f"Index manifest at {path} is unreadable: {exc}") from exc

        version = data.get("format_version")
        if version != FORMAT_VERSION:
            raise IndexMismatchError(
                f"Index at {store_dir} uses manifest format {version}, but this version of "
                f"EchoVector understands format {FORMAT_VERSION}. Re-index the store."
            )

        try:
            return cls(
                format_version=int(data["format_version"]),
                backend=str(data["backend"]),
                embedding_dim=int(data["embedding_dim"]),
                chunk_seconds=float(data["chunk_seconds"]),
                overlap_seconds=float(data["overlap_seconds"]),
                sample_rate=int(data["sample_rate"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexMismatchError(f"Index manifest at {path} is missing fields: {exc}") from exc

    def write(self, store_dir: str | Path) -> None:
        """Persist this manifest into a store directory.

        Args:
            store_dir: The index store directory. Created if missing.
        """
        path = self.path_for(store_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def assert_compatible_with(self, other: "IndexManifest", store_dir: str | Path) -> None:
        """Check that vectors written under `other` can be searched under `self`.

        Chunking settings are deliberately not fatal: they change how *new*
        audio is sliced but leave existing vectors searchable. Only the
        embedding space itself has to match.

        Args:
            self: Settings recorded when the index was built.
            other: Settings requested for the current session.
            store_dir: Store directory, used for the error message.

        Raises:
            IndexMismatchError: If the embedding spaces differ.
        """
        if self.embedding_dim != other.embedding_dim:
            raise IndexMismatchError(
                f"Index at {store_dir} holds {self.embedding_dim}-dimensional vectors, but the "
                f"requested backend produces {other.embedding_dim}-dimensional vectors. "
                f"Use a different --store-dir, or re-index with --reset."
            )
        if self.backend != other.backend:
            raise IndexMismatchError(
                f"Index at {store_dir} was built with backend '{self.backend}', but this session "
                f"requested '{other.backend}'. Scores across different models are meaningless. "
                f"Use a different --store-dir, or re-index with --reset."
            )

    def chunking_differences(self, other: "IndexManifest") -> list[str]:
        """Describe chunking settings that differ from another manifest.

        Args:
            other: Settings requested for the current session.

        Returns:
            Human-readable descriptions of each difference; empty if identical.
        """
        fields = ("chunk_seconds", "overlap_seconds", "sample_rate")
        return [
            f"{name} (indexed with {getattr(self, name)}, now {getattr(other, name)})"
            for name in fields
            if getattr(self, name) != getattr(other, name)
        ]
