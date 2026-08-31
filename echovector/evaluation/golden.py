"""Golden-set format, loader, and time-aware relevance judgement.

A golden set is a JSON document pairing natural-language queries with the audio
moments a human considers relevant::

    {
      "name": "smoke-set",
      "queries": [
        {
          "query": "dog barking",
          "relevant": [{"filepath": "a.wav", "start": 12.0, "end": 20.0}]
        }
      ]
    }

Search returns chunks whose boundaries were chosen by the chunker, not by the
annotator, so relevance is judged by temporal overlap rather than by equality.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from echovector.search.results import SearchResult

OverlapMode = Literal["shorter", "span", "result", "iou"]
"""How an overlap in seconds is turned into a fraction.

``shorter`` divides by the shorter of the two intervals, ``span`` by the
annotated span, ``result`` by the returned chunk, and ``iou`` by their union.
"""

_OVERLAP_MODES: frozenset[str] = frozenset({"shorter", "span", "result", "iou"})


class GoldenSetError(ValueError):
    """Raised when a golden set is malformed."""


@dataclass(frozen=True)
class RelevantSpan:
    """A human-annotated moment that a query should retrieve.

    Attributes:
        filepath: Path of the audio file containing the moment.
        start: Start time in seconds.
        end: End time in seconds, never before start.
    """

    filepath: str
    start: float
    end: float

    def __post_init__(self) -> None:
        """Validate the span.

        Raises:
            GoldenSetError: If the filepath is empty or the times are invalid.
        """
        if not self.filepath:
            raise GoldenSetError("Span filepath must not be empty.")
        if self.start < 0:
            raise GoldenSetError(f"Span start must be non-negative, got {self.start}.")
        if self.end < self.start:
            raise GoldenSetError(f"Span end {self.end} is before start {self.start}.")

    @property
    def duration(self) -> float:
        """Length of the span in seconds."""
        return self.end - self.start


@dataclass(frozen=True)
class GoldenQuery:
    """A query and the moments it is expected to retrieve.

    Attributes:
        query: Natural-language query text.
        relevant: Known-relevant spans, at least one.
        query_id: Optional stable identifier for reporting.
    """

    query: str
    relevant: tuple[RelevantSpan, ...]
    query_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the query.

        Raises:
            GoldenSetError: If the query text or the span list is empty.
        """
        if not self.query.strip():
            raise GoldenSetError("Query text must not be empty.")
        if not self.relevant:
            raise GoldenSetError(f"Query {self.query!r} has no relevant spans.")

    @property
    def label(self) -> str:
        """Identifier to show in reports, falling back to the query text."""
        return self.query_id or self.query


@dataclass(frozen=True)
class GoldenSet:
    """A collection of golden queries.

    Attributes:
        queries: The queries to evaluate.
        name: Optional name for reporting.
    """

    queries: tuple[GoldenQuery, ...]
    name: str | None = None

    def __len__(self) -> int:
        """Return the number of queries."""
        return len(self.queries)

    def __iter__(self) -> Iterator[GoldenQuery]:
        """Iterate over the queries."""
        return iter(self.queries)


def _require_mapping(value: object, where: str) -> dict[str, Any]:
    """Check that a decoded JSON value is an object.

    Args:
        value: Decoded JSON value.
        where: Dotted path used in the error message.

    Returns:
        The value as a dictionary.

    Raises:
        GoldenSetError: If the value is not a JSON object.
    """
    if not isinstance(value, dict):
        raise GoldenSetError(f"{where} must be an object, got {type(value).__name__}.")
    return value


def _require_str(value: object, where: str) -> str:
    """Check that a decoded JSON value is a string.

    Args:
        value: Decoded JSON value.
        where: Dotted path used in the error message.

    Returns:
        The value as a string.

    Raises:
        GoldenSetError: If the value is not a string.
    """
    if not isinstance(value, str):
        raise GoldenSetError(f"{where} must be a string, got {type(value).__name__}.")
    return value


def _require_number(value: object, where: str) -> float:
    """Check that a decoded JSON value is a number.

    Args:
        value: Decoded JSON value.
        where: Dotted path used in the error message.

    Returns:
        The value as a float.

    Raises:
        GoldenSetError: If the value is not a number. JSON booleans are rejected
            even though Python treats them as integers.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise GoldenSetError(f"{where} must be a number, got {type(value).__name__}.")
    return float(value)


def _parse_span(raw: object, where: str) -> RelevantSpan:
    """Build a relevant span from a decoded JSON object.

    Args:
        raw: Decoded JSON value for the span.
        where: Dotted path used in error messages.

    Returns:
        The parsed span.

    Raises:
        GoldenSetError: If a field is missing or has the wrong type.
    """
    data = _require_mapping(raw, where)
    for key in ("filepath", "start", "end"):
        if key not in data:
            raise GoldenSetError(f"{where} is missing required field {key!r}.")
    filepath = _require_str(data["filepath"], f"{where}.filepath")
    start = _require_number(data["start"], f"{where}.start")
    end = _require_number(data["end"], f"{where}.end")
    try:
        return RelevantSpan(filepath=filepath, start=start, end=end)
    except GoldenSetError as exc:
        raise GoldenSetError(f"{where}: {exc}") from exc


def _parse_query(raw: object, where: str) -> GoldenQuery:
    """Build a golden query from a decoded JSON object.

    Args:
        raw: Decoded JSON value for the query.
        where: Dotted path used in error messages.

    Returns:
        The parsed query.

    Raises:
        GoldenSetError: If a field is missing or has the wrong type.
    """
    data = _require_mapping(raw, where)
    if "query" not in data:
        raise GoldenSetError(f"{where} is missing required field 'query'.")
    if "relevant" not in data:
        raise GoldenSetError(f"{where} is missing required field 'relevant'.")

    relevant = data["relevant"]
    if not isinstance(relevant, list):
        raise GoldenSetError(f"{where}.relevant must be an array, got {type(relevant).__name__}.")
    if not relevant:
        raise GoldenSetError(f"{where}.relevant must not be empty.")

    query_id = data.get("id")
    if query_id is not None:
        query_id = _require_str(query_id, f"{where}.id")

    spans = tuple(
        _parse_span(item, f"{where}.relevant[{index}]") for index, item in enumerate(relevant)
    )
    query_text = _require_str(data["query"], f"{where}.query")
    try:
        return GoldenQuery(query=query_text, relevant=spans, query_id=query_id)
    except GoldenSetError as exc:
        raise GoldenSetError(f"{where}: {exc}") from exc


def parse_golden_set(data: object) -> GoldenSet:
    """Build a golden set from decoded JSON.

    Args:
        data: Decoded JSON document.

    Returns:
        The parsed golden set.

    Raises:
        GoldenSetError: If the document does not match the golden-set schema.
    """
    document = _require_mapping(data, "golden set")
    if "queries" not in document:
        raise GoldenSetError("Golden set is missing required field 'queries'.")

    queries = document["queries"]
    if not isinstance(queries, list):
        raise GoldenSetError(f"'queries' must be an array, got {type(queries).__name__}.")
    if not queries:
        raise GoldenSetError("Golden set must contain at least one query.")

    name = document.get("name")
    if name is not None:
        name = _require_str(name, "name")

    parsed = tuple(_parse_query(item, f"queries[{index}]") for index, item in enumerate(queries))
    return GoldenSet(queries=parsed, name=name)


def load_golden_set(path: str | Path) -> GoldenSet:
    """Load and validate a golden set from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed golden set.

    Raises:
        GoldenSetError: If the file is unreadable, is not valid JSON, or does not
            match the golden-set schema.
    """
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GoldenSetError(f"Cannot read golden set {file_path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GoldenSetError(f"Golden set {file_path} is not valid JSON: {exc}") from exc

    try:
        return parse_golden_set(data)
    except GoldenSetError as exc:
        raise GoldenSetError(f"Golden set {file_path} is invalid: {exc}") from exc


def _normalize_filepath(filepath: str, *, basename_only: bool) -> str:
    """Normalise a filepath for comparison.

    Args:
        filepath: Path as written in the golden set or returned by search.
        basename_only: Compare only the final path component.

    Returns:
        The normalised path.
    """
    normalized = os.path.normpath(filepath)
    return os.path.basename(normalized) if basename_only else normalized


@dataclass(frozen=True)
class RelevanceJudge:
    """Decides whether a search result covers an annotated span.

    A result is relevant when its filepath matches the span's and the two time
    ranges overlap by at least ``min_overlap_fraction`` of the interval selected
    by ``mode``. The default accepts any strictly positive overlap, so a chunk
    that merely touches a span at a single instant does not count.

    Attributes:
        min_overlap_fraction: Required overlap fraction in [0.0, 1.0].
        mode: Which interval the overlap fraction is measured against.
        match_basename: Compare only file names, for golden sets written with
            paths relative to a different root than the index.
    """

    min_overlap_fraction: float = 0.0
    mode: OverlapMode = "shorter"
    match_basename: bool = False

    def __post_init__(self) -> None:
        """Validate the judge configuration.

        Raises:
            ValueError: If the fraction is outside [0.0, 1.0] or the mode is
                unknown.
        """
        if not 0.0 <= self.min_overlap_fraction <= 1.0:
            raise ValueError(
                f"min_overlap_fraction must be in [0.0, 1.0], got {self.min_overlap_fraction}."
            )
        if self.mode not in _OVERLAP_MODES:
            raise ValueError(f"Unknown overlap mode {self.mode!r}.")

    def overlap_fraction(self, result: SearchResult, span: RelevantSpan) -> float:
        """Measure how much of the configured interval the overlap covers.

        Args:
            result: A search result.
            span: An annotated relevant span.

        Returns:
            The overlap fraction in [0.0, 1.0]; 0.0 when the files differ or the
            ranges are disjoint. A zero-length interval scores 1.0 when the other
            range contains it, since it can never accumulate overlap seconds.
        """
        if _normalize_filepath(result.filepath, basename_only=self.match_basename) != (
            _normalize_filepath(span.filepath, basename_only=self.match_basename)
        ):
            return 0.0

        result_start = result.timestamp_range.start
        result_end = result.timestamp_range.end
        result_duration = result_end - result_start
        overlap = min(result_end, span.end) - max(result_start, span.start)

        if result_duration <= 0 or span.duration <= 0:
            return 1.0 if overlap >= 0 else 0.0
        if overlap <= 0:
            return 0.0

        denominators: dict[str, float] = {
            "shorter": min(result_duration, span.duration),
            "span": span.duration,
            "result": result_duration,
            "iou": result_duration + span.duration - overlap,
        }
        return min(overlap / denominators[self.mode], 1.0)

    def temporal_iou(self, result: SearchResult, span: RelevantSpan) -> float:
        """Return the intersection-over-union of a result and a span in time.

        Independent of ``mode``, because this answers a different question than
        relevance does: not "does this result find the event" but "how tightly
        does it fit it". A span that covers the event but runs three times too
        long is fully relevant and poorly bounded, and only IoU says so.

        Args:
            result: A search result.
            span: An annotated relevant span.

        Returns:
            IoU in [0.0, 1.0]; 0.0 when the files differ or the ranges are disjoint.
        """
        if _normalize_filepath(result.filepath, basename_only=self.match_basename) != (
            _normalize_filepath(span.filepath, basename_only=self.match_basename)
        ):
            return 0.0

        start = result.timestamp_range.start
        end = result.timestamp_range.end
        overlap = min(end, span.end) - max(start, span.start)
        if overlap <= 0:
            return 0.0

        union = max(end, span.end) - min(start, span.start)
        return min(overlap / union, 1.0) if union > 0 else 1.0

    def best_iou(self, result: SearchResult, spans: Sequence[RelevantSpan]) -> float:
        """Return the best temporal IoU between a result and any annotated span.

        Args:
            result: A search result.
            spans: The query's annotated spans.

        Returns:
            The highest IoU achieved, or 0.0 when nothing overlaps.
        """
        return max((self.temporal_iou(result, span) for span in spans), default=0.0)

    def matches(self, result: SearchResult, span: RelevantSpan) -> bool:
        """Judge a single result against a single span.

        Args:
            result: A search result.
            span: An annotated relevant span.

        Returns:
            True if the result covers the span well enough to count.
        """
        fraction = self.overlap_fraction(result, span)
        if self.min_overlap_fraction <= 0.0:
            return fraction > 0.0
        return fraction >= self.min_overlap_fraction

    def matched_spans(
        self, result: SearchResult, spans: Sequence[RelevantSpan]
    ) -> tuple[int, ...]:
        """Find every span a result covers.

        Args:
            result: A search result.
            spans: The query's relevant spans.

        Returns:
            Indices into ``spans`` of the spans the result covers.
        """
        return tuple(index for index, span in enumerate(spans) if self.matches(result, span))
