"""Tests for search results and filters."""

import pytest

from echovector.search.filters import SearchFilter
from echovector.search.results import SearchResult, TimestampRange


def test_timestamp_range_valid() -> None:
    rng = TimestampRange(start=1.0, end=2.5)
    assert rng.start == 1.0
    assert rng.end == 2.5


def test_timestamp_range_invalid() -> None:
    with pytest.raises(ValueError, match="negative"):
        TimestampRange(start=-1.0, end=2.0)
        
    with pytest.raises(ValueError, match="less than"):
        TimestampRange(start=2.0, end=1.0)


def test_search_result_creation() -> None:
    res = SearchResult(
        filepath="audio.wav",
        timestamp_range=TimestampRange(1.0, 3.0),
        score=0.95,
        metadata={"key": "value"}
    )
    assert res.filepath == "audio.wav"
    assert res.score == 0.95
    assert res.metadata == {"key": "value"}


def test_search_filter_filepaths() -> None:
    r1 = SearchResult("a.wav", TimestampRange(0, 1), 0.9)
    r2 = SearchResult("b.wav", TimestampRange(0, 1), 0.8)
    r3 = SearchResult("c.wav", TimestampRange(0, 1), 0.7)
    
    sf = SearchFilter(filepaths=["a.wav", "c.wav"])
    filtered = sf.apply([r1, r2, r3])
    
    assert len(filtered) == 2
    assert filtered[0].filepath == "a.wav"
    assert filtered[1].filepath == "c.wav"


def test_search_filter_min_score() -> None:
    r1 = SearchResult("a.wav", TimestampRange(0, 1), 0.9)
    r2 = SearchResult("b.wav", TimestampRange(0, 1), 0.8)
    r3 = SearchResult("c.wav", TimestampRange(0, 1), 0.7)
    
    sf = SearchFilter(min_score=0.75)
    filtered = sf.apply([r1, r2, r3])
    
    assert len(filtered) == 2
    assert filtered[0].filepath == "a.wav"
    assert filtered[1].filepath == "b.wav"


def test_search_filter_metadata() -> None:
    r1 = SearchResult("a.wav", TimestampRange(0, 1), 0.9, metadata={"lang": "en"})
    r2 = SearchResult("b.wav", TimestampRange(0, 1), 0.8, metadata={"lang": "es"})
    r3 = SearchResult("c.wav", TimestampRange(0, 1), 0.7, metadata={"lang": "en", "author": "john"})
    
    sf = SearchFilter(metadata_filters={"lang": "en"})
    filtered = sf.apply([r1, r2, r3])
    
    assert len(filtered) == 2
    assert filtered[0].filepath == "a.wav"
    assert filtered[1].filepath == "c.wav"
    
    sf2 = SearchFilter(metadata_filters={"lang": "en", "author": "john"})
    filtered2 = sf2.apply([r1, r2, r3])
    
    assert len(filtered2) == 1
    assert filtered2[0].filepath == "c.wav"
