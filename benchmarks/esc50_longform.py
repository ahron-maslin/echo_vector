"""Measure chunking and span merging on long recordings built from ESC-50.

ESC-50 clips are uniform five-second single-source recordings, so on that corpus
every clip is exactly one chunk: chunk length never varies and span merging never
fires. Neither can be evaluated there.

This builds a long-form corpus instead, by concatenating clips into multi-minute
files. Because we place each clip ourselves, the ground truth is exact — the
span of "a dog barking" is known to the sample — which makes it possible to ask
two questions the fixed-grid design has only ever answered by argument:

1. Does chunk length matter, and where is the sweet spot? A 10s chunk over 5s
   clips always straddles two unrelated sounds; a 5s chunk aligns with them.
   Real recordings are never aligned, so the interesting number is how much the
   misalignment costs.
2. Does span merging help or hurt? It is on by default and gated by a relevance
   band that was chosen by reasoning rather than measurement.

Run::

    python benchmarks/esc50_longform.py --dataset ./ESC-50-master --backend clap
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.esc50 import CLIP_SECONDS, phrase_for, read_metadata
from echovector import EchoVector, SearchResult
from echovector.evaluation import evaluate, load_golden_set

#: Rate the concatenated recordings are written at.
SAMPLE_RATE = 44_100


def build_longform_corpus(
    dataset: Path,
    workdir: Path,
    clips_per_class: int,
    clips_per_file: int,
    seed: int,
) -> tuple[Path, Path, dict[str, Any]]:
    """Concatenate ESC-50 clips into long recordings with known ground truth.

    Clips are shuffled before grouping so that a file holds unrelated sounds
    back to back, which is what makes chunk misalignment bite.

    Args:
        dataset: Root of the extracted ESC-50 archive.
        workdir: Directory to build the corpus in.
        clips_per_class: How many clips of each class to place.
        clips_per_file: Clips concatenated into each long recording.
        seed: Shuffling seed, so a run is reproducible.

    Returns:
        The audio directory, the golden-set path, and a summary dictionary.
    """
    rows = read_metadata(dataset)
    source_audio = dataset / "audio"

    by_category: dict[str, list[str]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["filename"])

    rng = np.random.default_rng(seed)
    placements: list[tuple[str, str]] = []
    for category in sorted(by_category):
        chosen = sorted(by_category[category])[:clips_per_class]
        placements.extend((category, name) for name in chosen)
    rng.shuffle(placements)

    audio_dir = workdir / "audio"
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True)

    # category -> spans, accumulated as clips are placed into files.
    spans: dict[str, list[dict[str, Any]]] = {}
    file_count = 0

    for start in range(0, len(placements), clips_per_file):
        group = placements[start : start + clips_per_file]
        if not group:
            continue

        out_path = audio_dir / f"recording-{file_count:03d}.wav"
        buffers: list[np.ndarray] = []
        offset = 0.0
        for category, filename in group:
            audio, rate = sf.read(source_audio / filename, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.asarray(np.mean(audio, axis=1), dtype=np.float32)
            if rate != SAMPLE_RATE:  # pragma: no cover - ESC-50 is uniformly 44.1kHz
                raise SystemExit(f"Unexpected sample rate {rate} in {filename}")

            buffers.append(audio)
            spans.setdefault(category, []).append(
                {
                    "filepath": str(out_path.resolve()),
                    "start": offset,
                    "end": offset + CLIP_SECONDS,
                }
            )
            offset += len(audio) / SAMPLE_RATE

        sf.write(out_path, np.concatenate(buffers), SAMPLE_RATE, subtype="FLOAT")
        file_count += 1

    queries = [
        {"id": category, "query": phrase_for(category), "relevant": spans[category]}
        for category in sorted(spans)
    ]
    golden_path = workdir / "golden-longform.json"
    golden_path.write_text(
        json.dumps({"name": "esc50-longform", "queries": queries}, indent=2), encoding="utf-8"
    )

    summary = {
        "files": file_count,
        "clips": len(placements),
        "clips_per_file": clips_per_file,
        "file_seconds": clips_per_file * CLIP_SECONDS,
    }
    return audio_dir, golden_path, summary


def score_chunk_length(
    audio_dir: Path,
    golden_path: Path,
    workdir: Path,
    backend: str,
    chunk_seconds: float,
    ks: tuple[int, ...],
) -> dict[bool, dict[str, float]]:
    """Index at one chunk length, then score with merging off and on.

    Merging is a query-time option, so both settings are scored against a single
    index rather than paying for CLAP inference twice.

    Args:
        audio_dir: The long-form corpus.
        golden_path: Golden set describing where each sound sits.
        workdir: Directory to hold the index.
        backend: Embedding backend name.
        chunk_seconds: Chunk length to index at.
        ks: Cut-offs to score at.

    Returns:
        A mapping of merge_spans to that setting's "metric@k" values.
    """
    store_dir = workdir / f"index-{backend}-{chunk_seconds:g}"
    if store_dir.exists():
        shutil.rmtree(store_dir)

    engine = EchoVector(
        store_dir=store_dir,
        backend=backend,
        chunk_seconds=chunk_seconds,
        overlap_seconds=0.0,
    )
    engine.index(audio_dir)
    golden_set = load_golden_set(golden_path)

    scored: dict[bool, dict[str, float]] = {}
    for merge_spans in (False, True):

        def search(query: str, k: int, merge: bool = merge_spans) -> list[SearchResult]:
            return engine.search(query, top_k=k, merge_spans=merge)

        report = evaluate(search, golden_set, ks=ks)
        scored[merge_spans] = {
            f"{metric}@{k}": report.metric(metric, k)
            for k in ks
            for metric in ("recall", "precision", "ndcg", "mrr")
        }

    engine.close()
    return scored


def main() -> int:
    """Sweep chunk length with merging on and off, and print a comparison.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Extracted ESC-50 root directory.")
    parser.add_argument("--workdir", default="./benchmark-longform", help="Working directory.")
    parser.add_argument("--backend", default="clap", help="Embedding backend to evaluate.")
    parser.add_argument("--clips-per-class", type=int, default=6, help="Clips placed per class.")
    parser.add_argument("--clips-per-file", type=int, default=12, help="Clips per recording.")
    parser.add_argument("--chunk-seconds", default="2,5,10", help="Chunk lengths to sweep.")
    parser.add_argument("--ks", default="5,10", help="Comma-separated cut-offs.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffling seed.")
    parser.add_argument("--json-out", default="", help="Write all rows as JSON here.")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    audio_dir, golden_path, summary = build_longform_corpus(
        Path(args.dataset).resolve(),
        workdir,
        args.clips_per_class,
        args.clips_per_file,
        args.seed,
    )
    print(
        f"{summary['files']} recordings of {summary['file_seconds']:g}s "
        f"({summary['clips']} clips placed)\n"
    )

    ks = tuple(int(part) for part in args.ks.split(",") if part.strip())
    lengths = [float(part) for part in args.chunk_seconds.split(",") if part.strip()]
    metric_names = [f"{m}@{k}" for k in ks for m in ("recall", "precision", "ndcg", "mrr")]

    header = f"{'chunk':>6} {'merge':>6} " + " ".join(f"{name:>12}" for name in metric_names)
    print(header)
    print("-" * len(header))

    rows: list[dict[str, Any]] = []
    for chunk_seconds in lengths:
        scored = score_chunk_length(
            audio_dir, golden_path, workdir, args.backend, chunk_seconds, ks
        )
        for merge_spans, values in scored.items():
            rows.append({"chunk_seconds": chunk_seconds, "merge_spans": merge_spans, **values})
            cells = " ".join(f"{values[name]:>12.3f}" for name in metric_names)
            print(f"{chunk_seconds:>6g} {merge_spans!s:>6} {cells}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"corpus": summary, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
