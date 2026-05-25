"""Tests for the search engine."""

from typing import Any

from echovector.search.engine import SearchEngine
from echovector.search.filters import SearchFilter


class MockEmbedder:
    """Mock embedder for testing."""
    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class MockIndex:
    """Mock index for testing."""
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.last_vector: list[float] | None = None
        self.last_top_k: int | None = None
        
    def search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        self.last_vector = vector
        self.last_top_k = top_k
        return self.results[:top_k]


def test_search_engine_basic() -> None:
    raw_results = [
        {"filepath": "a.wav", "start": 0.0, "end": 2.0, "score": 0.9, "metadata": {"lang": "en"}},
        {"filepath": "b.wav", "start": 1.0, "end": 3.0, "score": 0.8, "metadata": {}},
    ]
    index = MockIndex(raw_results)
    embedder = MockEmbedder()
    engine = SearchEngine(index=index, embedder=embedder)  # type: ignore
    
    results = engine.search("test query", top_k=2)
    
    assert len(results) == 2
    assert results[0].filepath == "a.wav"
    assert results[0].timestamp_range.start == 0.0
    assert results[0].timestamp_range.end == 2.0
    assert results[0].score == 0.9
    assert results[0].metadata == {"lang": "en"}
    
    assert results[1].filepath == "b.wav"
    assert results[1].metadata == {}


def test_search_engine_with_filters() -> None:
    raw_results = [
        {"filepath": "a.wav", "start": 0.0, "end": 2.0, "score": 0.9, "metadata": {"lang": "en"}},
        {"filepath": "b.wav", "start": 1.0, "end": 3.0, "score": 0.8, "metadata": {"lang": "es"}},
        {"filepath": "c.wav", "start": 2.0, "end": 4.0, "score": 0.95, "metadata": {"lang": "en"}},
    ]
    index = MockIndex(raw_results)
    embedder = MockEmbedder()
    engine = SearchEngine(index=index, embedder=embedder)  # type: ignore
    
    filters = SearchFilter(metadata_filters={"lang": "en"})
    # top_k is 1, so with filters it fetches 5 from index, gets all 3, filters to a and c, then slices to 1.
    results = engine.search("query", top_k=1, filters=filters)
    
    assert len(results) == 1
    assert results[0].filepath == "a.wav"
    
def test_search_engine_top_k_slice_after_filter() -> None:
    raw_results = [
        {"filepath": "a.wav", "start": 0.0, "end": 2.0, "score": 0.9, "metadata": {}},
        {"filepath": "b.wav", "start": 1.0, "end": 3.0, "score": 0.8, "metadata": {}},
        {"filepath": "c.wav", "start": 2.0, "end": 4.0, "score": 0.7, "metadata": {}},
    ]
    index = MockIndex(raw_results)
    embedder = MockEmbedder()
    engine = SearchEngine(index=index, embedder=embedder)  # type: ignore
    
    filters = SearchFilter(min_score=0.75)
    results = engine.search("query", top_k=2, filters=filters)
    
    assert len(results) == 2
    assert results[0].filepath == "a.wav"
    assert results[1].filepath == "b.wav"
