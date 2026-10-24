"""Recover terms whisper spells inconsistently, by how they sound rather than how they are spelled.

Whisper renders one Hebrew term several ways within a single file, so exact text search misses
most of them. Ranking vocabulary by IPA similarity, weighted by how often a rendering occurs,
puts whisper's habitual spelling first: a near-match seen 33 times is the term, one seen once
is usually coincidence.
"""

import math
import re
from collections import Counter
from collections.abc import Callable

from rapidfuzz.distance import Levenshtein

from echovector.speech.transcript import Word

IpaFn = Callable[[list[str]], dict[str, str]]


def clean(word: str) -> str:
    """Lowercase and strip everything but word characters and apostrophes."""
    return re.sub(r"[^\w']", "", word.lower())


def build_vocab(words: list[Word], max_ngram: int = 2) -> Counter[str]:
    """Count every unigram and bigram surface form in a transcript."""
    cleaned = [c for c in (clean(w.text) for w in words) if c]
    vocab: Counter[str] = Counter()
    for size in range(1, max_ngram + 1):
        for i in range(len(cleaned) - size + 1):
            vocab[" ".join(cleaned[i : i + size])] += 1
    return vocab


def similarity(a: str, b: str) -> float:
    """Normalised edit similarity in [0, 1]."""
    if not a or not b:
        return 0.0
    return float(1.0 - Levenshtein.distance(a, b) / max(len(a), len(b)))


def expand_query(
    query: str,
    vocab: dict[str, int],
    ipa: IpaFn,
    query_ipa: str | None = None,
    top_k: int = 3,
    floor: float = 0.6,
) -> list[str]:
    """Return the query followed by the vocabulary entries that sound most like it."""
    base = clean(query) or query
    if query_ipa is None:
        query_ipa = ipa([query]).get(query, "")
    sounds = ipa(list(vocab))
    scored: list[tuple[float, str]] = []
    for surface, count in vocab.items():
        if surface == base:
            continue
        sim = similarity(query_ipa, sounds.get(surface, ""))
        if sim >= floor:
            scored.append((sim * (1.0 + math.log1p(count)), surface))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [query, *(surface for _, surface in scored[: max(top_k - 1, 0)])]
