"""Measure audio-to-audio retrieval ("find more that sounds like this") on ESC-50.

Text search depends on CLAP's text tower being well aligned with its audio tower.
Similarity search does not: it compares audio to audio, so it works with encoders
that have no text side at all. That makes it a separate capability worth its own
number, and the fallback if text retrieval ever disappoints.

Each clip in turn becomes the query, and a result counts as correct when it comes
from the same ESC-50 class. The query clip itself is excluded, so with N clips per
class there are N-1 correct answers available.

Run::

    python benchmarks/esc50_similar.py --dataset ./ESC-50-master --backend clap
    python benchmarks/esc50_similar.py --dataset ./ESC-50-master --backend wav2vec2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.esc50 import build_corpus
from echovector import EchoVector
from echovector.evaluation.metrics import precision_at_k, reciprocal_rank


def category_of(filename: str) -> str:
    """Return the ESC-50 class encoded in a clip's filename.

    ESC-50 names clips ``{fold}-{clip}-{take}-{target}.wav`` where target is the
    numeric class, which is stable across the dataset.

    Args:
        filename: The clip's basename.

    Returns:
        The numeric class as a string.
    """
    return filename.rsplit("-", 1)[-1].removesuffix(".wav")


def main() -> int:
    """Index ESC-50, query it with each clip, and score same-class retrieval.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Extracted ESC-50 root directory.")
    parser.add_argument("--workdir", default="./benchmark-similar", help="Working directory.")
    parser.add_argument("--backend", default="clap", help="Embedding backend to evaluate.")
    parser.add_argument("--per-class", type=int, default=10, help="Clips per class.")
    parser.add_argument("--ks", default="1,5,10", help="Comma-separated cut-offs.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--json-out", default="", help="Write the results as JSON here.")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    ks = tuple(int(part) for part in args.ks.split(",") if part.strip())

    audio_dir, _, summary = build_corpus(
        Path(args.dataset).resolve(), workdir, args.per_class, args.seed
    )
    print(f"{summary['clips']} clips across {summary['classes']} classes")

    store_dir = workdir / f"index-{args.backend}"
    if store_dir.exists():
        shutil.rmtree(store_dir)

    engine = EchoVector(
        store_dir=store_dir, backend=args.backend, chunk_seconds=10.0, overlap_seconds=0.0
    )
    started = time.perf_counter()
    engine.index(audio_dir)
    print(f"indexed in {time.perf_counter() - started:.1f}s\n")

    clips = sorted(audio_dir.iterdir())
    per_k_precision: dict[int, list[float]] = {k: [] for k in ks}
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []

    for clip in clips:
        expected = category_of(clip.name)
        query_started = time.perf_counter()
        results = engine.search_by_audio(clip, top_k=max(ks))
        latencies.append(time.perf_counter() - query_started)

        relevance = [category_of(Path(r.filepath).name) == expected for r in results]
        for k in ks:
            per_k_precision[k].append(precision_at_k(relevance, k))
        reciprocal_ranks.append(reciprocal_rank(relevance))

    header = f"{'metric':<12}" + "".join(f"{'k=' + str(k):>9}" for k in ks)
    print(header)
    print("-" * len(header))
    precision_row = "".join(
        f"{sum(per_k_precision[k]) / len(per_k_precision[k]):>9.3f}" for k in ks
    )
    print(f"{'precision':<12}{precision_row}")
    mean_rr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"{'mrr':<12}{mean_rr:>9.3f}")

    chance = (args.per_class - 1) / (summary["clips"] - 1)
    mean_latency = sum(latencies) / len(latencies)
    print(f"\nchance precision: {chance:.3f} | mean query {mean_latency * 1000:.0f} ms")

    if args.json_out:
        payload: dict[str, Any] = {
            "backend": args.backend,
            "chance_precision": round(chance, 4),
            "mrr": round(mean_rr, 4),
            "mean_query_ms": round(mean_latency * 1000, 1),
            "precision": {
                f"@{k}": round(sum(per_k_precision[k]) / len(per_k_precision[k]), 4) for k in ks
            },
            **summary,
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
