"""Tests for merging adjacent search hits into spans."""

from typing import Any

import pytest

from echovector.search.results import SearchResult, TimestampRange
from echovector.search.spans import limit_per_file, merge_adjacent_spans


def chunk(
    filepath: str,
    start: float,
    end: float,
    score: float,
    chunk_id: int | None = None,
    **extra: Any,
) -> SearchResult:
    """Build a chunk-level result shaped like the indexer emits."""
    metadata: dict[str, Any] = {
        "filepath": filepath,
        "filename": filepath.rsplit("/", 1)[-1],
        "start": start,
        "end": end,
        "duration": 120.0,
        "sample_rate": 48000,
    }
    if chunk_id is not None:
        metadata["chunk_id"] = chunk_id
    metadata.update(extra)
    return SearchResult(
        filepath=filepath,
        timestamp_range=TimestampRange(start, end),
        score=score,
        metadata=metadata,
    )


def spans(results: list[SearchResult]) -> list[tuple[str, float, float]]:
    return [(r.filepath, r.timestamp_range.start, r.timestamp_range.end) for r in results]


# --------------------------------------------------------------------------
# Totality: degenerate inputs
# --------------------------------------------------------------------------


def test_empty_input_returns_empty_list() -> None:
    assert merge_adjacent_spans([]) == []


def test_single_result_keeps_range_and_score() -> None:
    merged = merge_adjacent_spans([chunk("a.wav", 10.0, 20.0, 0.9, chunk_id=1)])

    assert len(merged) == 1
    assert merged[0].timestamp_range == TimestampRange(10.0, 20.0)
    assert merged[0].score == 0.9
    assert merged[0].metadata["merged_chunk_count"] == 1
    assert merged[0].metadata["chunk_ids"] == [1]


def test_disjoint_results_are_left_alone() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("a.wav", 60.0, 70.0, 0.8, chunk_id=6),
    ]

    assert spans(merge_adjacent_spans(results)) == [("a.wav", 0.0, 10.0), ("a.wav", 60.0, 70.0)]


def test_zero_length_ranges_merge_when_touching() -> None:
    results = [
        chunk("a.wav", 5.0, 5.0, 0.7, chunk_id=0),
        chunk("a.wav", 5.0, 5.0, 0.9, chunk_id=1),
    ]

    merged = merge_adjacent_spans(results)

    assert spans(merged) == [("a.wav", 5.0, 5.0)]
    assert merged[0].score == 0.9


def test_zero_length_range_stays_separate_beyond_the_gap() -> None:
    results = [
        chunk("a.wav", 5.0, 5.0, 0.7, chunk_id=0),
        chunk("a.wav", 9.0, 9.0, 0.9, chunk_id=1),
    ]

    assert spans(merge_adjacent_spans(results)) == [("a.wav", 9.0, 9.0), ("a.wav", 5.0, 5.0)]


def test_metadata_free_results_are_handled() -> None:
    bare = SearchResult("a.wav", TimestampRange(0.0, 10.0), 0.5)
    other = SearchResult("a.wav", TimestampRange(9.0, 19.0), 0.6)

    merged = merge_adjacent_spans([bare, other])

    assert merged[0].metadata["chunk_ids"] == []
    assert merged[0].metadata["merged_chunk_count"] == 2


# --------------------------------------------------------------------------
# Merging behaviour
# --------------------------------------------------------------------------


def test_overlapping_chunks_collapse_into_one_span() -> None:
    """The 40s passage that motivated the module: five chunks, one moment."""
    results = [
        chunk("talk.wav", 0.0, 10.0, 0.71, chunk_id=0),
        chunk("talk.wav", 9.0, 19.0, 0.83, chunk_id=1),
        chunk("talk.wav", 18.0, 28.0, 0.79, chunk_id=2),
        chunk("talk.wav", 27.0, 37.0, 0.74, chunk_id=3),
        chunk("talk.wav", 36.0, 46.0, 0.70, chunk_id=4),
    ]

    merged = merge_adjacent_spans(results)

    assert spans(merged) == [("talk.wav", 0.0, 46.0)]
    assert merged[0].score == pytest.approx(0.83)
    assert merged[0].metadata["merged_chunk_count"] == 5
    assert merged[0].metadata["chunk_ids"] == [0, 1, 2, 3, 4]


def test_unsorted_input_is_ordered_before_merging() -> None:
    results = [
        chunk("a.wav", 27.0, 37.0, 0.6, chunk_id=3),
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("a.wav", 18.0, 28.0, 0.7, chunk_id=2),
        chunk("a.wav", 9.0, 19.0, 0.8, chunk_id=1),
    ]

    merged = merge_adjacent_spans(results)

    assert spans(merged) == [("a.wav", 0.0, 37.0)]
    assert merged[0].metadata["chunk_ids"] == [0, 1, 2, 3]


def test_contained_range_does_not_shrink_the_span() -> None:
    results = [
        chunk("a.wav", 0.0, 30.0, 0.9, chunk_id=0),
        chunk("a.wav", 5.0, 6.0, 0.8, chunk_id=1),
        chunk("a.wav", 29.0, 31.0, 0.7, chunk_id=2),
    ]

    assert spans(merge_adjacent_spans(results)) == [("a.wav", 0.0, 31.0)]


def test_touching_ranges_merge_at_the_default_gap() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("a.wav", 10.0, 20.0, 0.8, chunk_id=1),
    ]

    assert spans(merge_adjacent_spans(results)) == [("a.wav", 0.0, 20.0)]


def test_gap_tolerance_controls_merging() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("a.wav", 12.0, 22.0, 0.8, chunk_id=1),
    ]

    assert len(merge_adjacent_spans(results)) == 2
    assert len(merge_adjacent_spans(results, max_gap_seconds=1.5)) == 2
    assert spans(merge_adjacent_spans(results, max_gap_seconds=2.0)) == [("a.wav", 0.0, 22.0)]


def test_files_are_never_merged_together() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("b.wav", 5.0, 15.0, 0.8, chunk_id=0),
        chunk("a.wav", 9.0, 19.0, 0.7, chunk_id=1),
        chunk("b.wav", 14.0, 24.0, 0.6, chunk_id=1),
    ]

    assert spans(merge_adjacent_spans(results)) == [("a.wav", 0.0, 19.0), ("b.wav", 5.0, 24.0)]


def test_many_files_interleaved_are_sorted_by_descending_score() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.5, chunk_id=0),
        chunk("b.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("c.wav", 0.0, 10.0, 0.7, chunk_id=0),
        chunk("a.wav", 9.0, 19.0, 0.6, chunk_id=1),
    ]

    merged = merge_adjacent_spans(results)

    assert [r.filepath for r in merged] == ["b.wav", "c.wav", "a.wav"]
    assert [r.score for r in merged] == [0.9, 0.7, 0.6]


def test_equal_scores_break_ties_deterministically() -> None:
    results = [
        chunk("b.wav", 30.0, 40.0, 0.5, chunk_id=3),
        chunk("a.wav", 30.0, 40.0, 0.5, chunk_id=3),
        chunk("a.wav", 0.0, 10.0, 0.5, chunk_id=0),
    ]

    assert spans(merge_adjacent_spans(results)) == [
        ("a.wav", 0.0, 10.0),
        ("a.wav", 30.0, 40.0),
        ("b.wav", 30.0, 40.0),
    ]


def test_input_results_are_not_mutated() -> None:
    original = chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0)
    snapshot = dict(original.metadata)

    merge_adjacent_spans([original, chunk("a.wav", 9.0, 19.0, 0.8, chunk_id=1)])

    assert original.metadata == snapshot
    assert original.timestamp_range == TimestampRange(0.0, 10.0)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_max_is_the_default_strategy() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.4, chunk_id=0),
        chunk("a.wav", 9.0, 19.0, 0.9, chunk_id=1),
    ]

    assert merge_adjacent_spans(results)[0].score == pytest.approx(0.9)
    assert merge_adjacent_spans(results, score_strategy="max")[0].score == pytest.approx(0.9)


def test_mean_strategy_averages_members() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.4, chunk_id=0),
        chunk("a.wav", 9.0, 19.0, 0.8, chunk_id=1),
    ]

    merged = merge_adjacent_spans(results, score_strategy="mean")

    assert merged[0].score == pytest.approx(0.6)
    assert merged[0].metadata["merged_score_strategy"] == "mean"


def test_long_mediocre_span_never_outranks_a_single_strong_hit() -> None:
    mediocre = [chunk("long.wav", 10.0 * i, 10.0 * i + 10.0, 0.3, chunk_id=i) for i in range(20)]
    strong = chunk("short.wav", 0.0, 10.0, 0.85, chunk_id=0)

    for strategy in ("max", "mean"):
        merged = merge_adjacent_spans([*mediocre, strong], score_strategy=strategy)  # type: ignore[arg-type]
        assert merged[0].filepath == "short.wav"


def test_unknown_score_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown score_strategy"):
        merge_adjacent_spans([], score_strategy="median")  # type: ignore[arg-type]


def test_negative_gap_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        merge_adjacent_spans([], max_gap_seconds=-1.0)


# --------------------------------------------------------------------------
# Metadata provenance
# --------------------------------------------------------------------------


def test_identical_member_fields_survive_the_merge() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("a.wav", 9.0, 19.0, 0.8, chunk_id=1),
    ]

    metadata = merge_adjacent_spans(results)[0].metadata

    assert metadata["filepath"] == "a.wav"
    assert metadata["filename"] == "a.wav"
    assert metadata["duration"] == 120.0
    assert metadata["sample_rate"] == 48000


def test_span_bounds_replace_member_bounds_in_metadata() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("a.wav", 9.0, 19.0, 0.8, chunk_id=1),
    ]

    metadata = merge_adjacent_spans(results)[0].metadata

    assert metadata["start"] == 0.0
    assert metadata["end"] == 19.0


def test_fields_that_differ_are_reported_not_dropped_silently() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0, speaker="alice"),
        chunk("a.wav", 9.0, 19.0, 0.8, chunk_id=1, speaker="bob"),
    ]

    metadata = merge_adjacent_spans(results)[0].metadata

    assert "speaker" not in metadata
    assert metadata["merged_divergent_keys"] == ["speaker"]


def test_recomputed_keys_are_not_listed_as_divergent() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("a.wav", 9.0, 19.0, 0.8, chunk_id=1),
    ]

    assert merge_adjacent_spans(results)[0].metadata["merged_divergent_keys"] == []


def test_a_field_missing_from_one_member_is_treated_as_divergent() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0, language="en"),
        chunk("a.wav", 9.0, 19.0, 0.8, chunk_id=1),
    ]

    metadata = merge_adjacent_spans(results)[0].metadata

    assert "language" not in metadata
    assert metadata["merged_divergent_keys"] == ["language"]


def test_merging_a_span_with_a_new_chunk_accumulates_provenance() -> None:
    first_pass = merge_adjacent_spans(
        [
            chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0, speaker="alice"),
            chunk("a.wav", 9.0, 19.0, 0.8, chunk_id=1, speaker="bob"),
        ]
    )
    second_pass = merge_adjacent_spans([*first_pass, chunk("a.wav", 18.0, 28.0, 0.7, chunk_id=2)])

    assert len(second_pass) == 1
    metadata = second_pass[0].metadata
    assert metadata["merged_chunk_count"] == 3
    assert metadata["chunk_ids"] == [0, 1, 2]
    # The divergence reported by the first pass is carried forward, not lost.
    assert metadata["merged_divergent_keys"] == ["speaker"]


# --------------------------------------------------------------------------
# Idempotence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["max", "mean"])
@pytest.mark.parametrize("gap", [0.0, 5.0])
def test_merging_is_idempotent(strategy: str, gap: float) -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.71, chunk_id=0),
        chunk("a.wav", 9.0, 19.0, 0.83, chunk_id=1),
        chunk("a.wav", 40.0, 50.0, 0.62, chunk_id=4),
        chunk("b.wav", 5.0, 15.0, 0.90, chunk_id=0, speaker="alice"),
        chunk("b.wav", 14.0, 24.0, 0.55, chunk_id=1, speaker="bob"),
        SearchResult("c.wav", TimestampRange(3.0, 3.0), 0.5),
    ]

    once = merge_adjacent_spans(results, max_gap_seconds=gap, score_strategy=strategy)  # type: ignore[arg-type]
    twice = merge_adjacent_spans(once, max_gap_seconds=gap, score_strategy=strategy)  # type: ignore[arg-type]

    assert twice == once


def test_idempotent_on_empty_and_single_inputs() -> None:
    assert merge_adjacent_spans(merge_adjacent_spans([])) == []

    once = merge_adjacent_spans([chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0)])
    assert merge_adjacent_spans(once) == once


# --------------------------------------------------------------------------
# limit_per_file
# --------------------------------------------------------------------------


def test_limit_per_file_caps_a_dominant_file() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("a.wav", 20.0, 30.0, 0.8, chunk_id=2),
        chunk("b.wav", 0.0, 10.0, 0.7, chunk_id=0),
        chunk("a.wav", 40.0, 50.0, 0.6, chunk_id=4),
        chunk("c.wav", 0.0, 10.0, 0.5, chunk_id=0),
    ]

    limited = limit_per_file(results, max_per_file=2)

    assert spans(limited) == [
        ("a.wav", 0.0, 10.0),
        ("a.wav", 20.0, 30.0),
        ("b.wav", 0.0, 10.0),
        ("c.wav", 0.0, 10.0),
    ]


def test_limit_per_file_preserves_relative_order() -> None:
    results = [
        chunk("b.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("a.wav", 0.0, 10.0, 0.8, chunk_id=0),
        chunk("b.wav", 20.0, 30.0, 0.7, chunk_id=2),
        chunk("a.wav", 20.0, 30.0, 0.6, chunk_id=2),
    ]

    limited = limit_per_file(results, max_per_file=1)

    assert [r.filepath for r in limited] == ["b.wav", "a.wav"]


def test_limit_per_file_is_a_no_op_when_the_cap_is_generous() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("b.wav", 0.0, 10.0, 0.8, chunk_id=0),
    ]

    assert limit_per_file(results, max_per_file=5) == results


def test_limit_per_file_on_empty_input() -> None:
    assert limit_per_file([], max_per_file=3) == []


def test_limit_per_file_does_not_mutate_its_input() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("a.wav", 20.0, 30.0, 0.8, chunk_id=2),
    ]

    limit_per_file(results, max_per_file=1)

    assert len(results) == 2


def test_limit_per_file_is_idempotent() -> None:
    results = [
        chunk("a.wav", 0.0, 10.0, 0.9, chunk_id=0),
        chunk("a.wav", 20.0, 30.0, 0.8, chunk_id=2),
        chunk("b.wav", 0.0, 10.0, 0.7, chunk_id=0),
    ]

    once = limit_per_file(results, max_per_file=1)
    assert limit_per_file(once, max_per_file=1) == once


def test_limit_per_file_rejects_a_non_positive_cap() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        limit_per_file([], max_per_file=0)
