"""Models for search results."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TimestampRange:
    """Represents a time range in an audio file.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        """Validate timestamp range."""
        if self.start < 0:
            raise ValueError("Start time cannot be negative.")
        if self.end < self.start:
            raise ValueError("End time cannot be less than start time.")


@dataclass(frozen=True)
class SearchResult:
    """A hydrated search result from the engine.

    Attributes:
        filepath: Path to the audio file.
        timestamp_range: The time range within the audio file.
        score: The search score (e.g., cosine similarity).
        metadata: Optional metadata dictionary.
    """

    filepath: str
    timestamp_range: TimestampRange
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
