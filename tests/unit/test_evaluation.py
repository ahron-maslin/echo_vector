"""Unit tests for echovector.evaluation: retrieval metrics, golden sets, and the harness."""

import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from echovector.evaluation.golden import (
    GoldenQuery,
    GoldenSet,
    GoldenSetError,
    RelevanceJudge,
    RelevantSpan,
    load_golden_set,
    parse_golden_set,
)
from echovector.evaluation.harness import evaluate, evaluate_query
from echovector.evaluation.metrics import (
    average_precision,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from echovector.search.results import SearchResult, TimestampRange


def result(filepath: str, start: float, end: float, score: float = 0.9) -> SearchResult:
    return SearchResult(
        filepath=filepath,
        timestamp_range=TimestampRange(start=start, end=end),
        score=score,
    )


# ---------------------------------------------------------------------------
# Ranked-retrieval metrics
# ---------------------------------------------------------------------------


def test_precision_at_k_counts_only_top_k():
    assert precision_at_k([True, False, True, True], 2) == pytest.approx(0.5)


def test_precision_at_k_divides_by_k_when_fewer_results():
    assert precision_at_k([True], 4) == pytest.approx(0.25)


def test_precision_at_k_empty_results():
    assert precision_at_k([], 5) == 0.0


@pytest.mark.parametrize("bad_k", [0, -1])
def test_metrics_reject_non_positive_k(bad_k):
    for call in (
        lambda: precision_at_k([True], bad_k),
        lambda: recall_at_k([True], 1, bad_k),
        lambda: hit_rate_at_k([True], bad_k),
        lambda: ndcg_at_k([1.0], bad_k),
    ):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            call()


def test_recall_at_k():
    assert recall_at_k([True, False, True], 4, 3) == pytest.approx(0.5)


def test_recall_at_k_no_relevant_items_is_zero_not_a_crash():
    assert recall_at_k([False], 0, 5) == 0.0


def test_recall_at_k_empty_results():
    assert recall_at_k([], 3, 5) == 0.0


def test_recall_at_k_rejects_negative_total():
    with pytest.raises(ValueError, match="non-negative"):
        recall_at_k([True], -1, 5)


def test_hit_rate_at_k():
    assert hit_rate_at_k([False, False, True], 3) == 1.0
    assert hit_rate_at_k([False, False, True], 2) == 0.0
    assert hit_rate_at_k([], 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([True]) == 1.0


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank([False, False]) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([[True], [False, True]]) == pytest.approx(0.75)


def test_mean_reciprocal_rank_no_queries():
    assert mean_reciprocal_rank([]) == 0.0


def test_average_precision_perfect_ranking():
    assert average_precision([True, True, False]) == pytest.approx(1.0)


def test_average_precision_interleaved():
    # Hits at ranks 1 and 3 -> (1/1 + 2/3) / 2.
    assert average_precision([True, False, True]) == pytest.approx((1.0 + 2 / 3) / 2)


def test_average_precision_explicit_total_penalises_misses():
    assert average_precision([True, False], total_relevant=4) == pytest.approx(0.25)


def test_average_precision_no_relevant_items():
    assert average_precision([False, False]) == 0.0
    assert average_precision([], total_relevant=0) == 0.0
    assert average_precision([True], total_relevant=0) == 0.0


def test_average_precision_rejects_negative_total():
    with pytest.raises(ValueError, match="non-negative"):
        average_precision([True], total_relevant=-2)


def test_ndcg_at_k_perfect_order_is_one():
    assert ndcg_at_k([1.0, 1.0, 0.0], 3) == pytest.approx(1.0)


def test_ndcg_at_k_penalises_bad_order():
    assert ndcg_at_k([0.0, 1.0], 2) == pytest.approx(1 / math.log2(3))


def test_ndcg_at_k_all_zero_gains():
    assert ndcg_at_k([0.0, 0.0], 2) == 0.0


def test_ndcg_at_k_empty_ranking():
    assert ndcg_at_k([], 5) == 0.0


def test_ndcg_at_k_ideal_gains_penalise_missing_items():
    # One hit retrieved but three were available: DCG 1.0 against an ideal of
    # 1 + 1/log2(3) + 1/log2(4).
    ideal = 1.0 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k([1.0], 3, ideal_gains=[1.0, 1.0, 1.0]) == pytest.approx(1.0 / ideal)


def test_ndcg_at_k_ideal_gains_all_zero():
    assert ndcg_at_k([1.0], 3, ideal_gains=[0.0]) == 0.0


# ---------------------------------------------------------------------------
# Golden set parsing and validation
# ---------------------------------------------------------------------------

VALID_DOCUMENT = {
    "name": "smoke",
    "queries": [
        {
            "id": "q1",
            "query": "dog barking",
            "relevant": [{"filepath": "a.wav", "start": 12.0, "end": 20.0}],
        },
        {
            "query": "glass breaking",
            "relevant": [
                {"filepath": "b.wav", "start": 0, "end": 5},
                {"filepath": "b.wav", "start": 30, "end": 33},
            ],
        },
    ],
}


def test_parse_golden_set_valid():
    golden = parse_golden_set(VALID_DOCUMENT)
    assert golden.name == "smoke"
    assert len(golden) == 2
    assert [q.label for q in golden] == ["q1", "glass breaking"]
    assert golden.queries[1].relevant[1] == RelevantSpan("b.wav", 30.0, 33.0)
    assert isinstance(golden.queries[1].relevant[0].start, float)


def test_load_golden_set_roundtrip(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(VALID_DOCUMENT), encoding="utf-8")
    assert load_golden_set(path) == parse_golden_set(VALID_DOCUMENT)


def test_load_golden_set_invalid_json(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="not valid JSON"):
        load_golden_set(path)


def test_load_golden_set_missing_file(tmp_path):
    with pytest.raises(GoldenSetError, match="Cannot read golden set"):
        load_golden_set(tmp_path / "nope.json")


def test_load_golden_set_reports_file_and_location(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"queries": [{"query": "x", "relevant": []}]}), encoding="utf-8")
    with pytest.raises(
        GoldenSetError, match=r"is invalid: queries\[0\].relevant must not be empty"
    ):
        load_golden_set(path)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "golden set must be an object"),
        ({}, "missing required field 'queries'"),
        ({"queries": {}}, "'queries' must be an array"),
        ({"queries": []}, "at least one query"),
        ({"queries": ["x"]}, r"queries\[0\] must be an object"),
        ({"queries": [{"relevant": []}]}, r"queries\[0\] is missing required field 'query'"),
        ({"queries": [{"query": "x"}]}, r"queries\[0\] is missing required field 'relevant'"),
        ({"queries": [{"query": "x", "relevant": {}}]}, r"queries\[0\].relevant must be an array"),
        (
            {"queries": [{"query": "x", "relevant": []}]},
            r"queries\[0\].relevant must not be empty",
        ),
        (
            {"queries": [{"query": 7, "relevant": [{"filepath": "a", "start": 0, "end": 1}]}]},
            r"queries\[0\].query must be a string",
        ),
        (
            {"queries": [{"query": "  ", "relevant": [{"filepath": "a", "start": 0, "end": 1}]}]},
            "Query text must not be empty",
        ),
        (
            {"queries": [{"query": "x", "relevant": [{"start": 0, "end": 1}]}]},
            r"queries\[0\].relevant\[0\] is missing required field 'filepath'",
        ),
        (
            {"queries": [{"query": "x", "relevant": [{"filepath": "a", "start": "0", "end": 1}]}]},
            r"queries\[0\].relevant\[0\].start must be a number",
        ),
        (
            {
                "queries": [
                    {"query": "x", "relevant": [{"filepath": "a", "start": True, "end": 1}]}
                ]
            },
            r"queries\[0\].relevant\[0\].start must be a number, got bool",
        ),
        (
            {"queries": [{"query": "x", "relevant": [{"filepath": "a", "start": -1, "end": 1}]}]},
            "Span start must be non-negative",
        ),
        (
            {"queries": [{"query": "x", "relevant": [{"filepath": "a", "start": 9, "end": 1}]}]},
            "is before start",
        ),
        (
            {"queries": [{"query": "x", "relevant": [{"filepath": "", "start": 0, "end": 1}]}]},
            "Span filepath must not be empty",
        ),
        (
            {
                "queries": [
                    {
                        "id": 3,
                        "query": "x",
                        "relevant": [{"filepath": "a", "start": 0, "end": 1}],
                    }
                ]
            },
            r"queries\[0\].id must be a string",
        ),
        ({"name": 1, "queries": [{"query": "x", "relevant": []}]}, "name must be a string"),
    ],
)
def test_parse_golden_set_rejects_malformed_input(document, message):
    with pytest.raises(GoldenSetError, match=message):
        parse_golden_set(document)


def test_golden_query_rejects_empty_spans():
    with pytest.raises(GoldenSetError, match="no relevant spans"):
        GoldenQuery(query="x", relevant=())


# ---------------------------------------------------------------------------
# Time-aware relevance judgement
# ---------------------------------------------------------------------------


def test_judge_default_accepts_any_overlap():
    judge = RelevanceJudge()
    span = RelevantSpan("a.wav", 12.0, 20.0)
    assert judge.matches(result("a.wav", 10.0, 13.0), span)
    assert judge.matches(result("a.wav", 19.5, 30.0), span)


def test_judge_rejects_disjoint_and_touching_ranges():
    judge = RelevanceJudge()
    span = RelevantSpan("a.wav", 12.0, 20.0)
    assert not judge.matches(result("a.wav", 0.0, 5.0), span)
    assert not judge.matches(result("a.wav", 0.0, 12.0), span)
    assert not judge.matches(result("a.wav", 20.0, 25.0), span)


def test_judge_requires_matching_filepath():
    judge = RelevanceJudge()
    assert not judge.matches(result("b.wav", 12.0, 20.0), RelevantSpan("a.wav", 12.0, 20.0))


def test_judge_normalizes_paths():
    judge = RelevanceJudge()
    assert judge.matches(
        result("./sounds/a.wav", 1.0, 2.0), RelevantSpan("sounds/a.wav", 1.0, 2.0)
    )


def test_judge_basename_matching_is_opt_in():
    span = RelevantSpan("data/a.wav", 1.0, 2.0)
    hit = result("/abs/other/a.wav", 1.0, 2.0)
    assert not RelevanceJudge().matches(hit, span)
    assert RelevanceJudge(match_basename=True).matches(hit, span)


def test_judge_overlap_fraction_modes():
    judge_shorter = RelevanceJudge(min_overlap_fraction=0.5)
    span = RelevantSpan("a.wav", 10.0, 12.0)
    chunk = result("a.wav", 0.0, 11.0)  # 1s overlap of a 2s span and an 11s chunk.

    assert judge_shorter.overlap_fraction(chunk, span) == pytest.approx(0.5)
    assert RelevanceJudge(mode="span").overlap_fraction(chunk, span) == pytest.approx(0.5)
    assert RelevanceJudge(mode="result").overlap_fraction(chunk, span) == pytest.approx(1 / 11)
    assert RelevanceJudge(mode="iou").overlap_fraction(chunk, span) == pytest.approx(1 / 12)


def test_judge_threshold_rejects_partial_overlap():
    span = RelevantSpan("a.wav", 10.0, 20.0)
    chunk = result("a.wav", 19.0, 30.0)
    assert RelevanceJudge().matches(chunk, span)
    assert not RelevanceJudge(min_overlap_fraction=0.5, mode="span").matches(chunk, span)
    assert RelevanceJudge(min_overlap_fraction=0.1, mode="span").matches(chunk, span)


def test_judge_fraction_never_exceeds_one():
    judge = RelevanceJudge()
    span = RelevantSpan("a.wav", 0.0, 10.0)
    assert judge.overlap_fraction(result("a.wav", 0.0, 10.0), span) == pytest.approx(1.0)


def test_judge_handles_zero_length_span():
    judge = RelevanceJudge(min_overlap_fraction=1.0)
    point = RelevantSpan("a.wav", 5.0, 5.0)
    assert judge.matches(result("a.wav", 0.0, 10.0), point)
    assert not judge.matches(result("a.wav", 6.0, 10.0), point)


def test_judge_handles_zero_length_result():
    judge = RelevanceJudge()
    span = RelevantSpan("a.wav", 0.0, 10.0)
    assert judge.matches(result("a.wav", 5.0, 5.0), span)
    assert not judge.matches(result("a.wav", 50.0, 50.0), span)


def test_judge_matched_spans_returns_all_indices():
    judge = RelevanceJudge()
    spans = (
        RelevantSpan("a.wav", 0.0, 5.0),
        RelevantSpan("a.wav", 4.0, 9.0),
        RelevantSpan("b.wav", 0.0, 5.0),
    )
    assert judge.matched_spans(result("a.wav", 4.5, 4.8), spans) == (0, 1)
    assert judge.matched_spans(result("a.wav", 90.0, 95.0), spans) == ()


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_judge_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="min_overlap_fraction"):
        RelevanceJudge(min_overlap_fraction=fraction)


def test_judge_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown overlap mode"):
        RelevanceJudge(mode="jaccard")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def make_golden(*queries: GoldenQuery, name: str | None = None) -> GoldenSet:
    return GoldenSet(queries=tuple(queries), name=name)


def test_evaluate_perfect_search():
    golden = make_golden(
        GoldenQuery("bark", (RelevantSpan("a.wav", 10.0, 15.0),)),
        name="tiny",
    )

    def search_fn(query, top_k):
        assert query == "bark"
        return [result("a.wav", 10.0, 20.0)][:top_k]

    report = evaluate(search_fn, golden, ks=(1, 5))
    assert report.name == "tiny"
    assert report.metric("recall", 1) == 1.0
    assert report.metric("precision", 1) == 1.0
    assert report.metric("hit_rate", 5) == 1.0
    assert report.metric("mrr", 5) == 1.0
    assert report.metric("ndcg", 1) == 1.0
    assert report.metric("map", 5) == 1.0


def test_evaluate_complete_miss():
    golden = make_golden(GoldenQuery("bark", (RelevantSpan("a.wav", 10.0, 15.0),)))

    def search_fn(query, top_k):
        return [result("b.wav", 0.0, 5.0), result("a.wav", 40.0, 50.0)][:top_k]

    report = evaluate(search_fn, golden, ks=(1, 2))
    for name in ("recall", "precision", "hit_rate", "mrr", "ndcg", "map"):
        assert report.metric(name, 2) == 0.0


def test_evaluate_empty_results_are_scored_not_crashed():
    golden = make_golden(GoldenQuery("bark", (RelevantSpan("a.wav", 10.0, 15.0),)))
    report = evaluate(lambda _query, _k: [], golden, ks=(1, 3))
    assert report.metric("recall", 3) == 0.0
    assert report.metric("precision", 1) == 0.0
    assert report.queries[0].num_retrieved == 0


def test_evaluate_recall_cannot_exceed_one_with_duplicate_span_hits():
    span = RelevantSpan("a.wav", 0.0, 30.0)
    golden = make_golden(GoldenQuery("hum", (span,)))

    def search_fn(query, top_k):
        return [result("a.wav", 0.0, 10.0), result("a.wav", 10.0, 20.0)][:top_k]

    report = evaluate(search_fn, golden, ks=(2,))
    assert report.metric("recall", 2) == 1.0
    assert report.metric("precision", 2) == 1.0


def test_evaluate_partial_recall_over_multiple_spans():
    golden = make_golden(
        GoldenQuery(
            "clap",
            (RelevantSpan("a.wav", 0.0, 5.0), RelevantSpan("a.wav", 60.0, 65.0)),
        )
    )

    def search_fn(query, top_k):
        return [result("a.wav", 1.0, 4.0), result("a.wav", 30.0, 35.0)][:top_k]

    report = evaluate(search_fn, golden, ks=(1, 2))
    assert report.metric("recall", 2) == pytest.approx(0.5)
    assert report.metric("precision", 2) == pytest.approx(0.5)
    assert report.metric("mrr", 2) == 1.0


def test_evaluate_rank_position_matters():
    golden = make_golden(GoldenQuery("bark", (RelevantSpan("a.wav", 10.0, 15.0),)))

    def search_fn(query, top_k):
        return [result("b.wav", 0.0, 5.0), result("a.wav", 10.0, 15.0)][:top_k]

    report = evaluate(search_fn, golden, ks=(1, 2))
    assert report.metric("recall", 1) == 0.0
    assert report.metric("recall", 2) == 1.0
    assert report.metric("mrr", 2) == pytest.approx(0.5)
    assert report.metric("ndcg", 2) == pytest.approx(1 / math.log2(3))


def test_evaluate_requests_max_k_once_per_query():
    calls: list[tuple[str, int]] = []
    golden = make_golden(GoldenQuery("bark", (RelevantSpan("a.wav", 0.0, 1.0),)))

    def search_fn(query, top_k):
        calls.append((query, top_k))
        return []

    evaluate(search_fn, golden, ks=(10, 1, 5))
    assert calls == [("bark", 10)]


def test_evaluate_truncates_oversized_result_lists():
    golden = make_golden(GoldenQuery("bark", (RelevantSpan("a.wav", 90.0, 95.0),)))

    def search_fn(query, top_k):
        return [result("a.wav", float(i), float(i) + 1.0) for i in range(50)] + [
            result("a.wav", 90.0, 95.0)
        ]

    report = evaluate(search_fn, golden, ks=(3,))
    assert report.queries[0].num_retrieved == 3
    assert report.metric("hit_rate", 3) == 0.0


def test_evaluate_averages_across_queries():
    golden = make_golden(
        GoldenQuery("hit", (RelevantSpan("a.wav", 0.0, 5.0),)),
        GoldenQuery("miss", (RelevantSpan("z.wav", 0.0, 5.0),)),
    )

    def search_fn(query, top_k):
        return [result("a.wav", 0.0, 5.0)][:top_k]

    report = evaluate(search_fn, golden, ks=(1,))
    assert report.metric("hit_rate", 1) == pytest.approx(0.5)
    assert len(report.queries) == 2


def test_evaluate_uses_custom_judge():
    golden = make_golden(GoldenQuery("bark", (RelevantSpan("a.wav", 10.0, 20.0),)))

    def search_fn(query, top_k):
        return [result("a.wav", 19.0, 40.0)][:top_k]

    strict = RelevanceJudge(min_overlap_fraction=0.5, mode="span")
    assert evaluate(search_fn, golden, ks=(1,)).metric("hit_rate", 1) == 1.0
    assert evaluate(search_fn, golden, ks=(1,), judge=strict).metric("hit_rate", 1) == 0.0


def test_evaluate_normalizes_ks():
    golden = make_golden(GoldenQuery("bark", (RelevantSpan("a.wav", 0.0, 1.0),)))
    report = evaluate(lambda _q, _k: [], golden, ks=[5, 1, 5])
    assert report.ks == (1, 5)


@pytest.mark.parametrize("bad_ks", [(), (0,), (3, -1)])
def test_evaluate_rejects_invalid_ks(bad_ks):
    golden = make_golden(GoldenQuery("bark", (RelevantSpan("a.wav", 0.0, 1.0),)))
    with pytest.raises(ValueError):
        evaluate(lambda _q, _k: [], golden, ks=bad_ks)


def test_evaluate_empty_golden_set():
    report = evaluate(lambda _q, _k: [], GoldenSet(queries=()))
    assert report.queries == ()
    assert report.aggregates == {}
    assert report.latency.queries_per_second == 0.0
    assert "No queries evaluated." in report.format_table()


def test_evaluate_records_latency():
    golden = make_golden(
        GoldenQuery("a", (RelevantSpan("a.wav", 0.0, 1.0),)),
        GoldenQuery("b", (RelevantSpan("a.wav", 0.0, 1.0),)),
    )
    report = evaluate(lambda _q, _k: [], golden, ks=(1,))
    assert report.latency.mean_seconds >= 0.0
    assert report.latency.p95_seconds >= report.latency.p50_seconds
    assert report.latency.queries_per_second > 0.0


def test_evaluate_query_returns_labels():
    query = GoldenQuery("bark", (RelevantSpan("a.wav", 0.0, 10.0),), query_id="q7")

    def search_fn(_query, top_k):
        return [result("a.wav", 5.0, 6.0), result("b.wav", 0.0, 1.0)][:top_k]

    evaluation = evaluate_query(search_fn, query, ks=(2,))
    assert evaluation.label == "q7"
    assert evaluation.relevance == (True, False)
    assert evaluation.novel_relevance == (True, False)
    assert evaluation.metrics["rr@2"] == 1.0


def test_format_table_is_plain_readable_text():
    golden = make_golden(
        GoldenQuery("bark", (RelevantSpan("a.wav", 0.0, 5.0),), query_id="q1"),
        name="smoke",
    )

    def search_fn(query, top_k):
        return [result("a.wav", 0.0, 5.0)][:top_k]

    table = evaluate(search_fn, golden, ks=(1, 5)).format_table(include_per_query=True)
    assert "smoke" in table
    assert "k=1" in table
    assert "k=5" in table
    assert "mrr" in table
    assert "map" in table
    assert "hit_rate" in table
    assert "queries/sec" in table
    assert "q1" in table
    assert "\x1b[" not in table


def test_report_to_dict_is_json_serializable():
    golden = make_golden(GoldenQuery("bark", (RelevantSpan("a.wav", 0.0, 5.0),)))
    report = evaluate(lambda _q, _k: [result("a.wav", 1.0, 2.0)], golden, ks=(1,))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["num_queries"] == 1
    assert payload["aggregates"]["recall@1"] == 1.0
    assert payload["queries"][0]["metrics"]["ap@1"] == 1.0


def test_report_metric_unknown_key():
    golden = make_golden(GoldenQuery("bark", (RelevantSpan("a.wav", 0.0, 5.0),)))
    report = evaluate(lambda _q, _k: [], golden, ks=(1,))
    with pytest.raises(KeyError):
        report.metric("recall", 99)


def test_evaluate_propagates_search_errors():
    golden = make_golden(GoldenQuery("bark", (RelevantSpan("a.wav", 0.0, 5.0),)))

    def search_fn(_query, _top_k):
        raise RuntimeError("index unavailable")

    with pytest.raises(RuntimeError, match="index unavailable"):
        evaluate(search_fn, golden, ks=(1,))


# ---------------------------------------------------------------------------
# Normalized-metric invariant: every score must stay within [0.0, 1.0]
# ---------------------------------------------------------------------------

NORMALIZED_METRICS = ("recall", "precision", "hit_rate", "mrr", "ndcg", "map")


def scenario(num_spans: int, targets):
    """Build a query with `num_spans` disjoint spans and results aimed at them.

    Each target selects the span a returned chunk overlaps; -1 returns a chunk
    that overlaps nothing. Repeating a target produces duplicate coverage of one
    span, which is what a passage split into adjacent chunks looks like.
    """
    spans = tuple(RelevantSpan("a.wav", 10.0 * i, 10.0 * i + 5.0) for i in range(num_spans))
    golden_query = GoldenQuery("q", spans)
    miss_start = 10.0 * num_spans + 100.0

    def search_fn(_query, top_k):
        results = []
        for index, target in enumerate(targets):
            if 0 <= target < num_spans:
                results.append(result("a.wav", 10.0 * target + 1.0, 10.0 * target + 3.0))
            else:
                results.append(
                    result("a.wav", miss_start + 5.0 * index, miss_start + 5.0 * index + 2.0)
                )
        return results[:top_k]

    return golden_query, search_fn


def assert_metrics_in_unit_interval(report):
    for k in report.ks:
        for name in NORMALIZED_METRICS:
            value = report.metric(name, k)
            assert 0.0 <= value <= 1.0, f"aggregate {name}@{k} = {value}"
    for evaluation in report.queries:
        for key, value in evaluation.metrics.items():
            assert 0.0 <= value <= 1.0, f"{evaluation.label} {key} = {value}"


@pytest.mark.parametrize(
    ("num_spans", "targets", "ks"),
    [
        # Regression: more relevant results than spans, all covering one span.
        (3, [0, 0, 0, 0, 0], (1, 3, 5)),
        # Duplicate coverage of one span alongside a genuine second hit.
        (2, [0, 0, 0, 1], (1, 2, 5)),
        # More spans than k.
        (5, [0, 1, 2, 3, 4], (1, 2)),
        # k larger than the result count.
        (2, [0], (1, 5, 10)),
        # Nothing retrieved at all.
        (2, [], (1, 5)),
        # Nothing relevant retrieved.
        (2, [-1, -1, -1], (1, 3)),
        # Single span, single perfect hit.
        (1, [0], (1,)),
    ],
)
def test_normalized_metrics_stay_in_unit_interval_for_known_shapes(num_spans, targets, ks):
    golden_query, search_fn = scenario(num_spans, targets)
    assert_metrics_in_unit_interval(evaluate(search_fn, make_golden(golden_query), ks=ks))


@given(
    num_spans=st.integers(min_value=1, max_value=4),
    targets=st.lists(st.integers(min_value=-1, max_value=5), max_size=9),
    ks=st.lists(st.integers(min_value=1, max_value=7), min_size=1, max_size=3, unique=True),
)
@settings(max_examples=300, deadline=None)
def test_normalized_metrics_stay_in_unit_interval(num_spans, targets, ks):
    resolved = [target if target < 0 else target % num_spans for target in targets]
    golden_query, search_fn = scenario(num_spans, resolved)
    assert_metrics_in_unit_interval(evaluate(search_fn, make_golden(golden_query), ks=ks))


def test_duplicate_span_coverage_does_not_inflate_ndcg_or_map():
    # Five adjacent chunks all covering the first of three spans: one span found.
    golden_query, search_fn = scenario(3, [0, 0, 0, 0, 0])
    report = evaluate(search_fn, make_golden(golden_query), ks=(5,))

    ideal = 1.0 + 1 / math.log2(3) + 1 / math.log2(4)
    assert report.metric("ndcg", 5) == pytest.approx(1.0 / ideal)
    assert report.metric("map", 5) == pytest.approx(1 / 3)
    assert report.metric("recall", 5) == pytest.approx(1 / 3)
    # Every returned chunk really is relevant, so precision stays at 1.0.
    assert report.metric("precision", 5) == 1.0
    assert report.metric("mrr", 5) == 1.0


def test_ranking_metrics_reward_covering_more_distinct_spans():
    duplicated, duplicated_fn = scenario(3, [0, 0, 0])
    distinct, distinct_fn = scenario(3, [0, 1, 2])
    duplicated_report = evaluate(duplicated_fn, make_golden(duplicated), ks=(3,))
    distinct_report = evaluate(distinct_fn, make_golden(distinct), ks=(3,))

    assert distinct_report.metric("ndcg", 3) == 1.0
    assert distinct_report.metric("map", 3) == 1.0
    assert duplicated_report.metric("ndcg", 3) < distinct_report.metric("ndcg", 3)
    assert duplicated_report.metric("map", 3) < distinct_report.metric("map", 3)


# ---------------------------------------------------------------------------
# Span fit (temporal IoU)
# ---------------------------------------------------------------------------


class TestTemporalIoU:
    """IoU asks how tightly a result bounds the event, not just whether it found it."""

    def _span(self, start: float, end: float, filepath: str = "/a.wav") -> RelevantSpan:
        return RelevantSpan(filepath=filepath, start=start, end=end)

    def _result(self, start: float, end: float, filepath: str = "/a.wav") -> SearchResult:
        return SearchResult(
            filepath=filepath,
            timestamp_range=TimestampRange(start=start, end=end),
            score=1.0,
        )

    def test_exact_match_scores_one(self) -> None:
        judge = RelevanceJudge()
        assert judge.temporal_iou(self._result(10.0, 20.0), self._span(10.0, 20.0)) == 1.0

    def test_disjoint_scores_zero(self) -> None:
        judge = RelevanceJudge()
        assert judge.temporal_iou(self._result(0.0, 5.0), self._span(10.0, 20.0)) == 0.0

    def test_different_file_scores_zero(self) -> None:
        judge = RelevanceJudge()
        result = self._result(10.0, 20.0, filepath="/other.wav")
        assert judge.temporal_iou(result, self._span(10.0, 20.0)) == 0.0

    def test_an_overlong_span_is_penalised_though_fully_relevant(self) -> None:
        """The case ranking metrics cannot see: right event, sloppy bounds."""
        judge = RelevanceJudge()
        span = self._span(10.0, 20.0)
        tight = self._result(10.0, 20.0)
        loose = self._result(0.0, 60.0)

        # Both fully cover the event, so both are relevant...
        assert judge.matches(tight, span) is True
        assert judge.matches(loose, span) is True
        # ...but only IoU distinguishes them.
        assert judge.temporal_iou(tight, span) == 1.0
        assert judge.temporal_iou(loose, span) == pytest.approx(10.0 / 60.0)

    def test_partial_overlap(self) -> None:
        judge = RelevanceJudge()
        # overlap 5s (10-15), union 15s (5-20)
        iou = judge.temporal_iou(self._result(5.0, 15.0), self._span(10.0, 20.0))
        assert iou == pytest.approx(5.0 / 15.0)

    def test_best_iou_picks_the_closest_span(self) -> None:
        judge = RelevanceJudge()
        spans = [self._span(0.0, 5.0), self._span(10.0, 20.0), self._span(50.0, 55.0)]
        assert judge.best_iou(self._result(10.0, 20.0), spans) == 1.0

    def test_best_iou_with_no_spans_is_zero(self) -> None:
        assert RelevanceJudge().best_iou(self._result(0.0, 1.0), []) == 0.0

    def test_report_exposes_span_fit(self) -> None:
        golden = parse_golden_set(
            {
                "queries": [
                    {
                        "query": "q",
                        "relevant": [{"filepath": "/a.wav", "start": 10.0, "end": 20.0}],
                    }
                ]
            }
        )

        def tight(_query: str, _k: int) -> list[SearchResult]:
            return [self._result(10.0, 20.0)]

        def loose(_query: str, _k: int) -> list[SearchResult]:
            return [self._result(0.0, 60.0)]

        tight_report = evaluate(tight, golden, ks=(1,))
        loose_report = evaluate(loose, golden, ks=(1,))

        # Identical on every ranking metric...
        assert tight_report.metric("recall", 1) == loose_report.metric("recall", 1) == 1.0
        # ...and separated only by span fit.
        assert tight_report.span_iou() == 1.0
        assert loose_report.span_iou() == pytest.approx(10.0 / 60.0)
        assert "Span fit" in tight_report.format_table()
        assert tight_report.to_dict()["aggregates"]["iou"] == 1.0

    def test_span_iou_of_an_empty_report_is_zero(self) -> None:
        # parse_golden_set rejects an empty set, so build one directly.
        report = evaluate(lambda _q, _k: [], GoldenSet(queries=()), ks=(1,))
        assert report.span_iou() == 0.0

    def test_results_that_find_nothing_score_zero_fit(self) -> None:
        golden = parse_golden_set(
            {
                "queries": [
                    {
                        "query": "q",
                        "relevant": [{"filepath": "/a.wav", "start": 10.0, "end": 20.0}],
                    }
                ]
            }
        )
        report = evaluate(lambda _q, _k: [self._result(90.0, 95.0)], golden, ks=(1,))
        assert report.span_iou() == 0.0
