"""Filtering logic for search results."""

from typing import Any

from echovector.search.results import SearchResult


class SearchFilter:
    """Filter parameters for search queries.

    Attributes:
        filepaths: Optional list of allowed file paths.
        min_score: Optional minimum score threshold.
        metadata_filters: Optional exact match metadata filters.
    """

    def __init__(
        self,
        filepaths: list[str] | None = None,
        min_score: float | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the search filter.

        Args:
            filepaths: List of valid file paths.
            min_score: Minimum required score.
            metadata_filters: Key-value pairs that must match exactly.
        """
        self.filepaths = filepaths
        self.min_score = min_score
        self.metadata_filters = metadata_filters or {}

    def apply(self, results: list[SearchResult]) -> list[SearchResult]:
        """Apply filters to a list of results.

        Args:
            results: List of SearchResult objects.

        Returns:
            Filtered list of SearchResult objects.
        """
        filtered = results
        if self.min_score is not None:
            filtered = [r for r in filtered if r.score >= self.min_score]

        if self.filepaths is not None:
            valid_paths = set(self.filepaths)
            filtered = [r for r in filtered if r.filepath in valid_paths]

        if self.metadata_filters:
            for key, val in self.metadata_filters.items():
                filtered = [r for r in filtered if r.metadata and r.metadata.get(key) == val]

        return filtered
