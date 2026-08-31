"""Retrieval evaluation: golden sets, ranked-retrieval metrics, and a harness."""

from echovector.evaluation.golden import (
    GoldenQuery,
    GoldenSet,
    GoldenSetError,
    OverlapMode,
    RelevanceJudge,
    RelevantSpan,
    load_golden_set,
    parse_golden_set,
)
from echovector.evaluation.harness import (
    DEFAULT_KS,
    EvaluationReport,
    LatencyStats,
    QueryEvaluation,
    SearchFn,
    evaluate,
    evaluate_query,
)
from echovector.evaluation.metrics import (
    average_precision,
    cosine_similarity,
    euclidean_distance,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "DEFAULT_KS",
    "EvaluationReport",
    "GoldenQuery",
    "GoldenSet",
    "GoldenSetError",
    "LatencyStats",
    "OverlapMode",
    "QueryEvaluation",
    "RelevanceJudge",
    "RelevantSpan",
    "SearchFn",
    "average_precision",
    "cosine_similarity",
    "euclidean_distance",
    "evaluate",
    "evaluate_query",
    "hit_rate_at_k",
    "load_golden_set",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "parse_golden_set",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
