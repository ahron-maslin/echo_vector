"""Vector similarity and ranked-retrieval evaluation metrics.

The retrieval metrics operate on binary relevance labels ordered by rank, which
keeps them independent of how relevance was judged (see
:mod:`echovector.evaluation.golden` for the time-aware judgement used by the
harness).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt


def cosine_similarity(vec1: npt.NDArray[np.float32], vec2: npt.NDArray[np.float32]) -> float:
    """Calculate the cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between -1.0 and 1.0.
    """
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape.")

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    dot_product = np.dot(vec1.flatten(), vec2.flatten())
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def euclidean_distance(vec1: npt.NDArray[np.float32], vec2: npt.NDArray[np.float32]) -> float:
    """Calculate the Euclidean distance between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Euclidean distance.
    """
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape.")

    distance = np.linalg.norm(vec1.flatten() - vec2.flatten())
    return float(distance)


def _require_positive_k(k: int) -> None:
    """Reject cut-offs that cannot define a ranking prefix.

    Args:
        k: Requested rank cut-off.

    Raises:
        ValueError: If k is not a positive integer.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")


def _require_non_negative_total(total_relevant: int) -> None:
    """Reject impossible relevant-item counts.

    Args:
        total_relevant: Number of known-relevant items for the query.

    Raises:
        ValueError: If total_relevant is negative.
    """
    if total_relevant < 0:
        raise ValueError(f"total_relevant must be non-negative, got {total_relevant}.")


def precision_at_k(relevances: Sequence[bool], k: int) -> float:
    """Calculate the fraction of the top-k ranked items that are relevant.

    The denominator is always k, so a ranking that returns fewer than k items is
    penalised the same way an empty tail would be.

    Args:
        relevances: Relevance label per ranked item, best rank first.
        k: Rank cut-off.

    Returns:
        Precision in [0.0, 1.0]; 0.0 when nothing was retrieved.

    Raises:
        ValueError: If k is not a positive integer.
    """
    _require_positive_k(k)
    if not relevances:
        return 0.0
    return sum(1 for relevant in relevances[:k] if relevant) / k


def recall_at_k(relevances: Sequence[bool], total_relevant: int, k: int) -> float:
    """Calculate the fraction of known-relevant items found in the top k.

    Each label must mark a *distinct* relevant item; two ranked items that both
    match the same relevant item would otherwise push the score above 1.0.

    Args:
        relevances: Relevance label per ranked item, best rank first.
        total_relevant: Number of known-relevant items for the query.
        k: Rank cut-off.

    Returns:
        Recall in [0.0, 1.0]; 0.0 when the query has no relevant items, since
        recall is undefined there.

    Raises:
        ValueError: If k is not a positive integer or total_relevant is negative.
    """
    _require_positive_k(k)
    _require_non_negative_total(total_relevant)
    if total_relevant == 0 or not relevances:
        return 0.0
    return sum(1 for relevant in relevances[:k] if relevant) / total_relevant


def hit_rate_at_k(relevances: Sequence[bool], k: int) -> float:
    """Report whether at least one relevant item appears in the top k.

    Args:
        relevances: Relevance label per ranked item, best rank first.
        k: Rank cut-off.

    Returns:
        1.0 if the top k contains a relevant item, else 0.0.

    Raises:
        ValueError: If k is not a positive integer.
    """
    _require_positive_k(k)
    return 1.0 if any(relevances[:k]) else 0.0


def reciprocal_rank(relevances: Sequence[bool]) -> float:
    """Calculate the reciprocal of the rank of the first relevant item.

    Args:
        relevances: Relevance label per ranked item, best rank first.

    Returns:
        1 / rank of the first relevant item, or 0.0 if none is relevant.
    """
    for index, relevant in enumerate(relevances, start=1):
        if relevant:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(relevances_per_query: Sequence[Sequence[bool]]) -> float:
    """Calculate the mean reciprocal rank across several queries.

    Args:
        relevances_per_query: One sequence of relevance labels per query.

    Returns:
        Mean reciprocal rank in [0.0, 1.0]; 0.0 when there are no queries.
    """
    if not relevances_per_query:
        return 0.0
    return sum(reciprocal_rank(labels) for labels in relevances_per_query) / len(
        relevances_per_query
    )


def average_precision(relevances: Sequence[bool], total_relevant: int | None = None) -> float:
    """Calculate average precision over a ranking.

    Args:
        relevances: Relevance label per ranked item, best rank first.
        total_relevant: Divisor for the summed precisions. Defaults to the number
            of relevant items present in ``relevances``. Pass ``min(R, k)`` to
            score a truncated ranking against an achievable ideal; the labels
            must then count the same items the divisor does, or the score can
            exceed 1.0.

    Returns:
        Average precision in [0.0, 1.0]; 0.0 when there is nothing relevant to
        find.

    Raises:
        ValueError: If total_relevant is negative.
    """
    if total_relevant is None:
        total_relevant = sum(1 for relevant in relevances if relevant)
    _require_non_negative_total(total_relevant)
    if total_relevant == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, relevant in enumerate(relevances, start=1):
        if relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / total_relevant


def _discounted_cumulative_gain(gains: Sequence[float]) -> float:
    """Sum rank-discounted gains.

    Args:
        gains: Gain per ranked item, best rank first.

    Returns:
        The discounted cumulative gain.
    """
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(gains: Sequence[float], k: int, ideal_gains: Sequence[float] | None = None) -> float:
    """Calculate normalised discounted cumulative gain at rank k.

    Args:
        gains: Gain per ranked item, best rank first. Binary relevance labels are
            accepted directly.
        k: Rank cut-off.
        ideal_gains: Gains available under a perfect ranking. Defaults to the
            retrieved gains re-sorted, which measures ordering quality only; pass
            the full set of known-relevant gains to also penalise misses. A
            ranking that scores gain the ideal does not account for exceeds 1.0,
            so both sequences must be drawn from the same pool of items.

    Returns:
        nDCG in [0.0, 1.0]; 0.0 when no gain is achievable.

    Raises:
        ValueError: If k is not a positive integer.
    """
    _require_positive_k(k)
    achievable = list(gains) if ideal_gains is None else list(ideal_gains)
    ideal = _discounted_cumulative_gain(sorted(achievable, reverse=True)[:k])
    if ideal <= 0:
        return 0.0
    return _discounted_cumulative_gain(list(gains)[:k]) / ideal
