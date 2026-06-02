"""Search engine implementation."""

from typing import Any, Protocol

from echovector.search.filters import SearchFilter
from echovector.search.results import SearchResult, TimestampRange


class Embedder(Protocol):
    """Protocol for text embedders."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a text query into a vector."""
        ...


class VectorIndex(Protocol):
    """Protocol for vector indices."""

    def search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """Search the index.

        Expected to return a list of dictionaries, each containing:
        - 'filepath': str
        - 'start': float
        - 'end': float
        - 'score': float
        - 'metadata': dict (optional)
        """
        ...


class SearchEngine:
    """Engine for executing searches against an index."""

    def __init__(self, index: VectorIndex, embedder: Embedder) -> None:
        """Initialize the search engine.

        Args:
            index: The vector index to search against.
            embedder: The embedder to use for queries.
        """
        self._index = index
        self._embedder = embedder

    def search(
        self, query: str, top_k: int = 10, filters: SearchFilter | None = None
    ) -> list[SearchResult]:
        """Search the index for a given query.

        Args:
            query: The text query.
            top_k: Number of results to return.
            filters: Optional filters to apply.

        Returns:
            A list of hydrated SearchResult objects.
        """
        vector = self._embedder.embed_text(query)

        # If filtering, fetch more to ensure we have enough post-filter
        fetch_k = top_k * 5 if filters else top_k

        raw_results = self._index.search(vector, fetch_k)

        results = []
        for raw in raw_results:
            # Safely get metadata
            metadata = raw.get("metadata", {})

            result = SearchResult(
                filepath=raw["filepath"],
                timestamp_range=TimestampRange(start=raw["start"], end=raw["end"]),
                score=raw["score"],
                metadata=metadata,
            )
            results.append(result)

        if filters:
            results = filters.apply(results)

        return results[:top_k]
