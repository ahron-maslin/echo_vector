"""Collapse chunk-level search hits into moment-level spans.

Indexing slices audio into short overlapping chunks, so one relevant passage
surfaces as several near-identical hits from the same file at consecutive
timestamps. The helpers here fold those hits back into the moments a user
actually asked for, and cap how much of a page any single file may occupy.

All functions are pure: inputs are never mutated and new frozen dataclasses are
constructed for every returned result.
"""

from collections.abc import Callable, Sequence
from statistics import fmean
from typing import Any, Literal

from echovector.search.results import SearchResult, TimestampRange

__all__ = ["ScoreStrategy", "limit_per_file", "merge_adjacent_spans"]

ScoreStrategy = Literal["max", "mean"]
"""How a merged span derives its score from the chunks that formed it."""

_CHUNK_ID_KEY = "chunk_id"
_CHUNK_IDS_KEY = "chunk_ids"
_COUNT_KEY = "merged_chunk_count"
_DIVERGENT_KEY = "merged_divergent_keys"
_STRATEGY_KEY = "merged_score_strategy"

# Keys the merge recomputes itself, so their divergence across members carries
# no information worth reporting back to the caller.
_RESET_KEYS = frozenset(
    {"start", "end", _CHUNK_ID_KEY, _CHUNK_IDS_KEY, _COUNT_KEY, _DIVERGENT_KEY, _STRATEGY_KEY}
)


def _max_score(scores: Sequence[float]) -> float:
    return max(scores)


def _mean_score(scores: Sequence[float]) -> float:
    return fmean(scores)


_SCORE_REDUCERS: dict[str, Callable[[Sequence[float]], float]] = {
    "max": _max_score,
    "mean": _mean_score,
}


def _relevance_threshold(group: Sequence[SearchResult], min_relative_score: float) -> float:
    """Return the score a hit must reach to be merged with its neighbours.

    Scored relative to the file's own best hit rather than an absolute value,
    because similarity scales differ across embedding backends.
    """
    if min_relative_score <= 0.0:
        return float("-inf")
    best = max(r.score for r in group)
    # A negative best would invert the comparison, so a relative band is
    # meaningless there; fall back to merging on adjacency alone.
    return best * min_relative_score if best > 0.0 else float("-inf")


def merge_adjacent_spans(
    results: list[SearchResult],
    max_gap_seconds: float = 0.0,
    score_strategy: ScoreStrategy = "max",
    min_relative_score: float = 0.0,
) -> list[SearchResult]:
    """Merge overlapping or near-adjacent hits from the same file into spans.

    Results are grouped by ``filepath``, ordered by start time, and folded
    together whenever the next hit starts no more than ``max_gap_seconds``
    after the running end of the current span. Overlapping and contained
    ranges always merge, whatever the gap allowance.

    ``score_strategy`` defaults to ``"max"``: the span inherits the score of
    its strongest chunk, so merging is rank-preserving -- a span lands exactly
    where its best chunk landed in the unmerged ranking, and a long span that
    accumulated many mediocre chunks can never outrank a single strong hit.
    ``"mean"`` is available for the opposite bias, rewarding spans that are
    uniformly relevant, but it lets an arbitrary chunk boundary dilute a strong
    hit whose neighbour happened to be weak.

    Merging is idempotent: re-merging an already-merged list with the same
    arguments returns an equal list.

    Args:
        results: Chunk-level results to fold. Not mutated.
        max_gap_seconds: Largest silence, in seconds, that two hits may be
            separated by and still belong to the same span. ``0.0`` merges only
            hits that overlap or touch exactly.
        score_strategy: How the span's score is derived from its members,
            either ``"max"`` or ``"mean"``.
        min_relative_score: A neighbour is absorbed only if it scores at least
            this fraction of the strongest hit in the file. Adjacency alone is
            not enough: a chunk that merely borders a strong hit can be
            entirely irrelevant, and merging it would widen the span to cover
            audio the user did not ask for. ``0.0`` (the default) merges purely
            on adjacency.

    Returns:
        Merged spans sorted by descending score, ties broken by filepath and
        start time. Each carries the provenance keys ``merged_chunk_count``,
        ``chunk_ids``, ``merged_score_strategy`` and ``merged_divergent_keys``
        in its metadata, alongside every field that was identical across the
        merged members and span-accurate ``start`` and ``end`` values.

    Raises:
        ValueError: If ``max_gap_seconds`` is negative or ``score_strategy``
            is not a recognised strategy.
    """
    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be non-negative.")
    if score_strategy not in _SCORE_REDUCERS:
        known = ", ".join(sorted(_SCORE_REDUCERS))
        raise ValueError(f"Unknown score_strategy {score_strategy!r}. Expected one of: {known}.")
    if min_relative_score < 0:
        raise ValueError("min_relative_score must be non-negative.")

    by_file: dict[str, list[SearchResult]] = {}
    for result in results:
        by_file.setdefault(result.filepath, []).append(result)

    merged: list[SearchResult] = []
    for group in by_file.values():
        threshold = _relevance_threshold(group, min_relative_score)
        ordered = sorted(group, key=lambda r: (r.timestamp_range.start, r.timestamp_range.end))
        run = [ordered[0]]
        run_end = ordered[0].timestamp_range.end
        for result in ordered[1:]:
            too_far = result.timestamp_range.start - run_end > max_gap_seconds
            too_weak = result.score < threshold or run[-1].score < threshold
            if too_far or too_weak:
                merged.append(_collapse(run, score_strategy))
                run = [result]
                run_end = result.timestamp_range.end
            else:
                run.append(result)
                run_end = max(run_end, result.timestamp_range.end)
        merged.append(_collapse(run, score_strategy))

    return sorted(
        merged,
        key=lambda r: (-r.score, r.filepath, r.timestamp_range.start, r.timestamp_range.end),
    )


def limit_per_file(results: list[SearchResult], max_per_file: int) -> list[SearchResult]:
    """Cap how many results a single file may contribute, keeping the best.

    Relative order is preserved, so applying this to a ranked list keeps that
    ranking and simply drops each file's surplus, lower-ranked entries.

    Args:
        results: Results to diversify, in the order they should be presented.
            Not mutated.
        max_per_file: Maximum number of results to keep per filepath.

    Returns:
        A new list holding at most ``max_per_file`` results per filepath, in
        the order they appeared in the input.

    Raises:
        ValueError: If ``max_per_file`` is less than one.
    """
    if max_per_file < 1:
        raise ValueError("max_per_file must be at least 1.")

    kept: list[SearchResult] = []
    counts: dict[str, int] = {}
    for result in results:
        count = counts.get(result.filepath, 0)
        if count >= max_per_file:
            continue
        counts[result.filepath] = count + 1
        kept.append(result)
    return kept


def _collapse(members: list[SearchResult], score_strategy: ScoreStrategy) -> SearchResult:
    """Fold one run of same-file results into a single span.

    Args:
        members: Results forming the span, ordered by start time.
        score_strategy: How the span's score is derived from its members.

    Returns:
        A new result spanning every member.
    """
    span = TimestampRange(
        start=min(member.timestamp_range.start for member in members),
        end=max(member.timestamp_range.end for member in members),
    )
    return SearchResult(
        filepath=members[0].filepath,
        timestamp_range=span,
        score=_SCORE_REDUCERS[score_strategy]([member.score for member in members]),
        metadata=_merge_metadata(members, span, score_strategy),
    )


def _merge_metadata(
    members: list[SearchResult],
    span: TimestampRange,
    score_strategy: ScoreStrategy,
) -> dict[str, Any]:
    """Build the merged span's metadata from its members' metadata.

    Args:
        members: Results forming the span, ordered by start time.
        span: The merged time range.
        score_strategy: How the span's score was derived.

    Returns:
        A new metadata mapping carrying every identical member field, the span
        bounds, and the span's provenance.
    """
    payloads = [member.metadata for member in members]
    shared = _shared_fields(payloads)

    merged = dict(shared)
    merged["start"] = span.start
    merged["end"] = span.end
    # Counts and chunk ids accumulate from the members rather than being
    # recounted, so a span that is merged again keeps its original provenance.
    merged[_COUNT_KEY] = sum(_member_count(payload) for payload in payloads)
    merged[_CHUNK_IDS_KEY] = _collect_chunk_ids(payloads)
    merged[_STRATEGY_KEY] = score_strategy
    merged[_DIVERGENT_KEY] = _divergent_keys(payloads, shared)
    return merged


def _shared_fields(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the fields every member agrees on.

    Args:
        payloads: Metadata mappings of the merged members.

    Returns:
        The key-value pairs present with an equal value in every mapping.
    """
    first, rest = payloads[0], payloads[1:]
    return {
        key: value
        for key, value in first.items()
        if all(key in other and other[key] == value for other in rest)
    }


def _divergent_keys(payloads: list[dict[str, Any]], shared: dict[str, Any]) -> list[str]:
    """List the member fields that the merge could not carry over verbatim.

    Args:
        payloads: Metadata mappings of the merged members.
        shared: The fields every member agreed on.

    Returns:
        Sorted key names, so nothing is dropped silently. Keys the merge
        recomputes (``start``, ``end``, ``chunk_id`` and the provenance keys)
        are excluded, and keys already reported by a previously merged member
        are carried forward.
    """
    dropped = {str(key) for payload in payloads for key in payload if key not in shared}
    inherited = {str(key) for payload in payloads for key in _as_list(payload.get(_DIVERGENT_KEY))}
    return sorted((dropped | inherited) - _RESET_KEYS)


def _collect_chunk_ids(payloads: list[dict[str, Any]]) -> list[Any]:
    """Gather the source chunk ids behind a span, in start order.

    Args:
        payloads: Metadata mappings of the merged members, in start order.

    Returns:
        Deduplicated chunk ids, flattening the ids of members that were
        themselves merged spans.
    """
    collected: list[Any] = []
    for payload in payloads:
        if _CHUNK_IDS_KEY in payload:
            candidates = _as_list(payload[_CHUNK_IDS_KEY])
        elif _CHUNK_ID_KEY in payload:
            candidates = [payload[_CHUNK_ID_KEY]]
        else:
            candidates = []
        for chunk_id in candidates:
            if chunk_id not in collected:
                collected.append(chunk_id)
    return collected


def _member_count(payload: dict[str, Any]) -> int:
    """Count how many chunks a member already represents.

    Args:
        payload: Metadata mapping of one merged member.

    Returns:
        The member's recorded chunk count, or one if it is an unmerged chunk.
    """
    raw = payload.get(_COUNT_KEY, 1)
    return int(raw) if isinstance(raw, int) else 1


def _as_list(value: Any) -> list[Any]:
    """Normalise a metadata value into a list.

    Args:
        value: A missing, scalar, or already-sequenced metadata value.

    Returns:
        The value as a list, empty when it is missing.
    """
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]
