"""Retrieval evaluation harness.

Runs a golden set against any search callable and reports ranked-retrieval
quality plus query latency. The harness never imports the engine, so the same
code evaluates :meth:`echovector.core.EchoVector.search`, a filtered search, or a
stub in tests.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from echovector.evaluation.golden import RelevanceJudge
from echovector.evaluation.metrics import (
    average_precision,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from echovector.evaluation.golden import GoldenQuery, GoldenSet, RelevantSpan
    from echovector.search.results import SearchResult

type SearchFn = Callable[[str, int], list["SearchResult"]]
"""A search entry point taking ``(query, top_k)`` and returning ranked results."""

DEFAULT_KS: tuple[int, ...] = (1, 5, 10)

_PER_QUERY_METRICS: tuple[str, ...] = ("recall", "precision", "hit_rate", "rr", "ndcg", "ap")
_AGGREGATE_NAMES: dict[str, str] = {"rr": "mrr", "ap": "map"}


@dataclass(frozen=True)
class QueryEvaluation:
    """Per-query outcome of one evaluation run.

    Attributes:
        query: The query text that was searched.
        label: Identifier used in reports.
        num_relevant_spans: Number of annotated spans for the query.
        num_retrieved: Number of results the search returned, capped at max(ks).
        relevance: Whether each ranked result covers any annotated span.
        novel_relevance: Whether each ranked result is the first to cover a span.
            Recall, nDCG, and average precision count these rather than
            ``relevance``, so duplicate hits on one span cannot push a normalised
            metric above 1.0.
        mean_iou: Average temporal IoU between each returned result and its
            best-matching annotated span. Measures how tightly results bound the
            events they found, which the ranking metrics do not ask about.
        latency_seconds: Wall-clock time spent inside the search callable.
        metrics: Metric name and cut-off (``"recall@5"``) mapped to its score.
    """

    query: str
    label: str
    num_relevant_spans: int
    num_retrieved: int
    relevance: tuple[bool, ...]
    novel_relevance: tuple[bool, ...]
    mean_iou: float
    latency_seconds: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LatencyStats:
    """Latency summary over the queries in a run.

    Attributes:
        mean_seconds: Mean query time.
        p50_seconds: Median query time (nearest-rank).
        p95_seconds: 95th percentile query time (nearest-rank).
        queries_per_second: Queries divided by the total time spent searching,
            excluding harness overhead.
    """

    mean_seconds: float
    p50_seconds: float
    p95_seconds: float
    queries_per_second: float


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate and per-query results of an evaluation run.

    Attributes:
        ks: Rank cut-offs evaluated, ascending.
        queries: Per-query outcomes in golden-set order.
        aggregates: Mean of each metric over all queries, keyed ``"recall@5"``.
        latency: Latency summary for the run.
        name: Optional golden-set name.
    """

    ks: tuple[int, ...]
    queries: tuple[QueryEvaluation, ...]
    aggregates: dict[str, float]
    latency: LatencyStats
    name: str | None = None

    def span_iou(self) -> float:
        """Return the mean temporal IoU of returned results with annotated spans.

        Ranking metrics say whether the right moment was found; this says how
        tightly the returned span bounds it. A result that covers the event but
        runs far too long scores fully on the former and poorly here.

        Returns:
            Mean IoU in [0.0, 1.0], or 0.0 when nothing was evaluated.
        """
        if not self.queries:
            return 0.0
        return sum(evaluation.mean_iou for evaluation in self.queries) / len(self.queries)

    def metric(self, name: str, k: int) -> float:
        """Look up one aggregate score.

        Args:
            name: Aggregate metric name, such as ``"recall"`` or ``"mrr"``.
            k: Rank cut-off.

        Returns:
            The aggregate score.

        Raises:
            KeyError: If the metric was not computed for that cut-off.
        """
        return self.aggregates[f"{name}@{k}"]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to JSON-serialisable primitives.

        Returns:
            A dictionary with the run's aggregates, latency, and per-query rows.
        """
        return {
            "name": self.name,
            "ks": list(self.ks),
            "num_queries": len(self.queries),
            "aggregates": dict(self.aggregates),
            "latency": {
                "mean_seconds": self.latency.mean_seconds,
                "p50_seconds": self.latency.p50_seconds,
                "p95_seconds": self.latency.p95_seconds,
                "queries_per_second": self.latency.queries_per_second,
            },
            "queries": [
                {
                    "query": evaluation.query,
                    "label": evaluation.label,
                    "num_relevant_spans": evaluation.num_relevant_spans,
                    "num_retrieved": evaluation.num_retrieved,
                    "latency_seconds": evaluation.latency_seconds,
                    "metrics": dict(evaluation.metrics),
                }
                for evaluation in self.queries
            ],
        }

    def format_table(self, include_per_query: bool = False) -> str:
        """Render the report as plain text.

        Args:
            include_per_query: Also render one row per query, showing each
                query's score at the largest cut-off.

        Returns:
            A printable multi-line string with no ANSI escapes or markup.
        """
        title = f"Retrieval evaluation{f' - {self.name}' if self.name else ''}"
        lines = [title, f"{len(self.queries)} queries, k = {', '.join(str(k) for k in self.ks)}"]
        if not self.queries:
            lines.append("No queries evaluated.")
            return "\n".join(lines)

        lines.append("")
        lines.extend(self._aggregate_rows())
        lines.append("")
        lines.append(f"Span fit (mean IoU with the annotated span): {self.span_iou():.3f}")
        lines.append(
            "Latency: mean "
            f"{self.latency.mean_seconds * 1000:.1f} ms | "
            f"p50 {self.latency.p50_seconds * 1000:.1f} ms | "
            f"p95 {self.latency.p95_seconds * 1000:.1f} ms | "
            f"{self.latency.queries_per_second:.1f} queries/sec"
        )
        if include_per_query:
            lines.append("")
            lines.extend(self._per_query_rows())
        return "\n".join(lines)

    def _aggregate_rows(self) -> list[str]:
        """Build the metric-by-cut-off table.

        Returns:
            Header and one row per metric.
        """
        names = [_AGGREGATE_NAMES.get(name, name) for name in _PER_QUERY_METRICS]
        name_width = max(len("metric"), *(len(name) for name in names))
        header = "  ".join(["metric".ljust(name_width), *(f"k={k}".rjust(7) for k in self.ks)])
        rows = [header, "-" * len(header)]
        for name in names:
            cells = [f"{self.aggregates.get(f'{name}@{k}', 0.0):.3f}".rjust(7) for k in self.ks]
            rows.append("  ".join([name.ljust(name_width), *cells]))
        return rows

    def _per_query_rows(self) -> list[str]:
        """Build the per-query table at the largest cut-off.

        Returns:
            Header and one row per query.
        """
        k = self.ks[-1]
        label_width = min(48, max(len("query"), *(len(q.label) for q in self.queries)))
        header = "  ".join(
            [
                "query".ljust(label_width),
                "hits".rjust(5),
                f"recall@{k}".rjust(10),
                f"ndcg@{k}".rjust(8),
                f"rr@{k}".rjust(6),
                "ms".rjust(7),
            ]
        )
        rows = [header, "-" * len(header)]
        for evaluation in self.queries:
            hits = sum(1 for relevant in evaluation.novel_relevance[:k] if relevant)
            rows.append(
                "  ".join(
                    [
                        evaluation.label[:label_width].ljust(label_width),
                        f"{hits}/{evaluation.num_relevant_spans}".rjust(5),
                        f"{evaluation.metrics.get(f'recall@{k}', 0.0):.3f}".rjust(10),
                        f"{evaluation.metrics.get(f'ndcg@{k}', 0.0):.3f}".rjust(8),
                        f"{evaluation.metrics.get(f'rr@{k}', 0.0):.3f}".rjust(6),
                        f"{evaluation.latency_seconds * 1000:.1f}".rjust(7),
                    ]
                )
            )
        return rows


def _normalize_ks(ks: Sequence[int]) -> tuple[int, ...]:
    """Deduplicate and sort the requested cut-offs.

    Args:
        ks: Requested rank cut-offs.

    Returns:
        Ascending unique cut-offs.

    Raises:
        ValueError: If no cut-off was given or any is not positive.
    """
    if not ks:
        raise ValueError("At least one k is required.")
    if any(k <= 0 for k in ks):
        raise ValueError(f"All k values must be positive integers, got {tuple(ks)}.")
    return tuple(sorted(set(ks)))


def _label_results(
    results: Sequence[SearchResult],
    spans: Sequence[RelevantSpan],
    judge: RelevanceJudge,
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Label ranked results against a query's annotated spans.

    Args:
        results: Ranked search results.
        spans: The query's relevant spans.
        judge: Time-aware relevance judge.

    Returns:
        Per-result relevance flags, and flags marking only the first result to
        cover each span.
    """
    covered: set[int] = set()
    relevance: list[bool] = []
    novel_relevance: list[bool] = []
    for result in results:
        matched = judge.matched_spans(result, spans)
        relevance.append(bool(matched))
        newly_covered = set(matched) - covered
        novel_relevance.append(bool(newly_covered))
        covered |= newly_covered
    return tuple(relevance), tuple(novel_relevance)


def _query_metrics(
    relevance: Sequence[bool],
    novel_relevance: Sequence[bool],
    num_spans: int,
    ks: Sequence[int],
) -> dict[str, float]:
    """Score one query at every cut-off.

    Recall, nDCG, and average precision are normalised against the *distinct*
    annotated spans, so their numerators count distinct spans covered
    (``novel_relevance``) rather than relevant results. Crediting every result
    that re-covers an already-found span would let the numerator exceed an ideal
    built from the span count, pushing nDCG and AP above 1.0; it would also mean
    a passage split into many adjacent relevant chunks outscored the same
    passage returned once, rewarding exactly the duplication span merging
    removes. These three therefore measure how quickly a ranking covers the
    distinct relevant material.

    Precision and hit rate keep per-result relevance: they are not normalised
    against the span count, and a chunk that re-covers a found span is still a
    result the user would call relevant. Reciprocal rank is unaffected either
    way, since the first relevant result is always the first novel one.

    Args:
        relevance: Per-result relevance flags.
        novel_relevance: Flags marking the first result to cover each span.
        num_spans: Number of annotated spans for the query.
        ks: Ascending rank cut-offs.

    Returns:
        Metric name and cut-off mapped to score, each in [0.0, 1.0].
    """
    metrics: dict[str, float] = {}
    for k in ks:
        # A ranking of k slots cannot surface more than k of the spans, so the
        # rank-sensitive metrics are normalised against that achievable ideal.
        achievable = min(num_spans, k)
        top_k_novel = list(novel_relevance[:k])
        metrics[f"recall@{k}"] = recall_at_k(novel_relevance, num_spans, k)
        metrics[f"precision@{k}"] = precision_at_k(relevance, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(relevance, k)
        metrics[f"rr@{k}"] = reciprocal_rank(list(relevance[:k]))
        metrics[f"ndcg@{k}"] = ndcg_at_k(
            [1.0 if relevant else 0.0 for relevant in top_k_novel],
            k,
            ideal_gains=[1.0] * achievable,
        )
        metrics[f"ap@{k}"] = average_precision(top_k_novel, total_relevant=achievable)
    return metrics


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile of a sample.

    Args:
        values: Sample values, in any order.
        fraction: Percentile as a fraction in [0.0, 1.0].

    Returns:
        The percentile, or 0.0 for an empty sample.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def _latency_stats(latencies: Sequence[float]) -> LatencyStats:
    """Summarise per-query latencies.

    Args:
        latencies: Wall-clock seconds per query.

    Returns:
        The latency summary; all zeros when no query ran.
    """
    if not latencies:
        return LatencyStats(0.0, 0.0, 0.0, 0.0)
    total = sum(latencies)
    return LatencyStats(
        mean_seconds=total / len(latencies),
        p50_seconds=_percentile(latencies, 0.50),
        p95_seconds=_percentile(latencies, 0.95),
        queries_per_second=len(latencies) / total if total > 0 else float("inf"),
    )


def _aggregate(evaluations: Sequence[QueryEvaluation], ks: Sequence[int]) -> dict[str, float]:
    """Average every per-query metric over the run.

    Args:
        evaluations: Per-query outcomes.
        ks: Ascending rank cut-offs.

    Returns:
        Aggregate metric name and cut-off mapped to mean score.
    """
    aggregates: dict[str, float] = {}
    if not evaluations:
        return aggregates
    # Not per-cut-off: IoU describes the shape of the results returned, not how
    # far down the ranking you look.
    aggregates["iou"] = sum(e.mean_iou for e in evaluations) / len(evaluations)
    for name in _PER_QUERY_METRICS:
        for k in ks:
            key = f"{name}@{k}"
            total = sum(evaluation.metrics[key] for evaluation in evaluations)
            aggregates[f"{_AGGREGATE_NAMES.get(name, name)}@{k}"] = total / len(evaluations)
    return aggregates


def evaluate_query(
    search_fn: SearchFn,
    golden_query: GoldenQuery,
    ks: Sequence[int] = DEFAULT_KS,
    judge: RelevanceJudge | None = None,
) -> QueryEvaluation:
    """Run and score a single golden query.

    Args:
        search_fn: Callable taking ``(query, top_k)`` and returning ranked results.
        golden_query: The query and its annotated spans.
        ks: Rank cut-offs to score.
        judge: Relevance judge; defaults to any-overlap matching.

    Returns:
        The per-query outcome.

    Raises:
        ValueError: If ``ks`` is empty or contains a non-positive cut-off.
    """
    cutoffs = _normalize_ks(ks)
    judge = judge or RelevanceJudge()
    max_k = cutoffs[-1]

    started = time.perf_counter()
    results = list(search_fn(golden_query.query, max_k))
    latency = time.perf_counter() - started

    # A search callable that ignores top_k must not earn credit past max(ks).
    results = results[:max_k]
    relevance, novel_relevance = _label_results(results, golden_query.relevant, judge)
    ious = [judge.best_iou(result, golden_query.relevant) for result in results]
    return QueryEvaluation(
        query=golden_query.query,
        label=golden_query.label,
        num_relevant_spans=len(golden_query.relevant),
        num_retrieved=len(results),
        relevance=relevance,
        novel_relevance=novel_relevance,
        mean_iou=sum(ious) / len(ious) if ious else 0.0,
        latency_seconds=latency,
        metrics=_query_metrics(relevance, novel_relevance, len(golden_query.relevant), cutoffs),
    )


def evaluate(
    search_fn: SearchFn,
    golden_set: GoldenSet,
    ks: Sequence[int] = DEFAULT_KS,
    judge: RelevanceJudge | None = None,
) -> EvaluationReport:
    """Evaluate a search callable against a golden set.

    Each query is searched once at ``max(ks)`` and scored at every cut-off, so
    the reported latency is one real query's cost rather than one per cut-off.

    Args:
        search_fn: Callable taking ``(query, top_k)`` and returning ranked
            results ordered best first.
        golden_set: Queries and their known-relevant spans.
        ks: Rank cut-offs to score.
        judge: Relevance judge; defaults to any-overlap matching.

    Returns:
        The evaluation report.

    Raises:
        ValueError: If ``ks`` is empty or contains a non-positive cut-off.
    """
    cutoffs = _normalize_ks(ks)
    judge = judge or RelevanceJudge()
    evaluations = tuple(
        evaluate_query(search_fn, golden_query, cutoffs, judge)
        for golden_query in golden_set.queries
    )
    return EvaluationReport(
        ks=cutoffs,
        queries=evaluations,
        aggregates=_aggregate(evaluations, cutoffs),
        latency=_latency_stats([evaluation.latency_seconds for evaluation in evaluations]),
        name=golden_set.name,
    )
