"""Concept search over transcripts: embed word windows, expand queries by sound."""

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from echovector.speech.phonetic import IpaFn, build_vocab, expand_query
from echovector.speech.transcript import Word, chunk_words

Embedder = Callable[..., npt.NDArray[np.float32]]


@dataclass(frozen=True)
class SpeechHit:
    """A passage in one file that matched a query."""

    file: str
    start: float
    end: float
    text: str
    score: float


class SpeechIndex:
    """A directory holding chunk vectors, their text and times, and the phonetic vocabulary.

    The embedder is called as `embedder(texts, query=bool)` and must return L2-normalised rows,
    so a dot product is cosine similarity. At shiur scale (tens of thousands of chunks) brute
    force over a numpy matrix is instant, so there is no ANN structure to maintain.
    """

    def __init__(
        self,
        directory: Path,
        embedder: Embedder,
        ipa: IpaFn,
        chunk_size: int = 45,
        stride: int = 25,
    ) -> None:
        self.directory = directory
        self._embedder = embedder
        self._ipa_backend = ipa
        self._chunk_size = chunk_size
        self._stride = stride
        self._rows: list[dict[str, Any]] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._vocab: dict[str, Counter[str]] = {}
        self._ipa_cache: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        meta = self.directory / "chunks.json"
        if not meta.exists():
            return
        self._rows = json.loads(meta.read_text())
        self._vectors = np.load(self.directory / "vectors.npy")
        vocab = json.loads((self.directory / "vocab.json").read_text())
        self._vocab = {file: Counter(counts) for file, counts in vocab.items()}
        self._ipa_cache = json.loads((self.directory / "ipa.json").read_text())

    def save(self) -> None:
        """Persist the index to its directory."""
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "chunks.json").write_text(json.dumps(self._rows))
        np.save(self.directory / "vectors.npy", self._vectors)
        (self.directory / "vocab.json").write_text(json.dumps(self._vocab))
        (self.directory / "ipa.json").write_text(json.dumps(self._ipa_cache))

    def _ipa(self, surfaces: list[str]) -> dict[str, str]:
        missing = sorted({s for s in surfaces if s not in self._ipa_cache})
        if missing:
            self._ipa_cache.update(self._ipa_backend(missing))
        return {s: self._ipa_cache.get(s, "") for s in surfaces}

    def add(self, file: str, words: list[Word]) -> None:
        """Index a transcript, replacing any earlier entry for the same file."""
        self._drop(file)
        chunks = chunk_words(words, self._chunk_size, self._stride)
        if not chunks:
            return
        vectors = self._embedder([c.text for c in chunks], query=False)
        self._vectors = vectors if self._vectors.size == 0 else np.vstack([self._vectors, vectors])
        self._rows.extend(
            {"file": file, "start": c.start, "end": c.end, "text": c.text} for c in chunks
        )
        vocab = build_vocab(words)
        self._vocab[file] = vocab
        self._ipa(list(vocab))

    def _drop(self, file: str) -> None:
        keep = [i for i, row in enumerate(self._rows) if row["file"] != file]
        if len(keep) != len(self._rows):
            self._rows = [self._rows[i] for i in keep]
            self._vectors = self._vectors[keep]
        self._vocab.pop(file, None)

    def search(self, query: str, k: int = 5, query_ipa: str | None = None) -> list[SpeechHit]:
        """Best passages for a query, scored by the closest of its phonetic variants."""
        if not self._rows:
            return []
        merged: Counter[str] = Counter()
        for counts in self._vocab.values():
            merged.update(counts)
        variants = expand_query(query, merged, self._ipa, query_ipa=query_ipa)
        scores = (self._vectors @ self._embedder(variants, query=True).T).max(axis=1)
        return [
            SpeechHit(
                file=self._rows[i]["file"],
                start=self._rows[i]["start"],
                end=self._rows[i]["end"],
                text=self._rows[i]["text"],
                score=float(scores[i]),
            )
            for i in np.argsort(-scores)[:k]
        ]
