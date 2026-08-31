"""Build an ESC-50 golden set and measure text-to-audio retrieval on it.

ESC-50 is 2000 five-second environmental recordings across 50 balanced classes
(dog, rain, chainsaw, crying baby, ...). Each class name is a natural-language
description of a sound, which makes it a direct test of the thing EchoVector
claims to do: find audio from a description of how it sounds.

Run it end to end::

    python benchmarks/esc50.py --dataset ./ESC-50-master --backend clap

Download the corpus first from https://github.com/karoldvl/ESC-50 (about 600 MB).

Two knobs matter for interpreting the output:

- ``--per-class`` samples the corpus down. With all 40 clips of a class relevant
  to its query, recall@10 cannot exceed 0.25 however perfect the ranking is, so
  the default samples 10 clips per class and recall@10 becomes reachable.
- ``--backend local`` gives the floor. The local backend knows nothing about
  semantics, so its scores are what "no retrieval ability" looks like on this
  set, and the CLAP numbers are only meaningful next to them.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echovector import EchoVector
from echovector.evaluation import evaluate, load_golden_set

#: ESC-50 clip length. Every clip is exactly this long.
CLIP_SECONDS = 5.0

#: Class names are stored slug-style ("crying_baby"); a handful read poorly when
#: naively de-slugged, so they get a phrasing closer to how someone would ask.
QUERY_OVERRIDES = {
    "clock_alarm": "an alarm clock ringing",
    "clock_tick": "a clock ticking",
    "can_opening": "a can being opened",
    "glass_breaking": "glass breaking",
    "vacuum_cleaner": "a vacuum cleaner running",
    "washing_machine": "a washing machine running",
    "water_drops": "water dripping",
    "pouring_water": "water being poured",
    "toilet_flush": "a toilet flushing",
    "brushing_teeth": "someone brushing their teeth",
    "drinking_sipping": "someone drinking and sipping",
    "mouse_click": "a mouse being clicked",
    "keyboard_typing": "someone typing on a keyboard",
    "door_wood_knock": "a knock on a wooden door",
    "door_wood_creaks": "a wooden door creaking",
    "crackling_fire": "a crackling fire",
    "chirping_birds": "birds chirping",
    "hand_saw": "a hand saw cutting wood",
    "church_bells": "church bells ringing",
    "car_horn": "a car horn honking",
    "engine": "an engine running",
    "train": "a train passing",
    "airplane": "an airplane flying overhead",
    "helicopter": "a helicopter flying",
    "crying_baby": "a baby crying",
    "laughing": "people laughing",
    "breathing": "someone breathing",
    "coughing": "someone coughing",
    "sneezing": "someone sneezing",
    "snoring": "someone snoring",
    "footsteps": "footsteps",
    "clapping": "people clapping",
    "thunderstorm": "thunder during a storm",
    "sea_waves": "sea waves on a shore",
}


def phrase_for(category: str) -> str:
    """Turn an ESC-50 class slug into a natural query.

    Args:
        category: The slug from the dataset metadata, e.g. "crying_baby".

    Returns:
        A description someone might plausibly type.
    """
    if category in QUERY_OVERRIDES:
        return QUERY_OVERRIDES[category]
    return f"a {category.replace('_', ' ')}"


def read_metadata(dataset: Path) -> list[dict[str, str]]:
    """Read ESC-50's metadata CSV.

    Args:
        dataset: Root of the extracted ESC-50 archive.

    Returns:
        One row per clip.

    Raises:
        SystemExit: If the dataset layout is not what ESC-50 ships.
    """
    meta_path = dataset / "meta" / "esc50.csv"
    if not meta_path.exists():
        raise SystemExit(
            f"No ESC-50 metadata at {meta_path}. Download and extract "
            f"https://github.com/karoldvl/ESC-50/archive/master.zip first."
        )
    with meta_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_corpus(
    dataset: Path,
    workdir: Path,
    per_class: int,
    seed: int,
) -> tuple[Path, Path, dict[str, Any]]:
    """Sample a balanced subset and write the audio plus a golden set.

    Args:
        dataset: Root of the extracted ESC-50 archive.
        workdir: Directory to build the corpus and golden set in.
        per_class: Clips to keep per class, or 0 to keep all 40.
        seed: Sampling seed, so a run is reproducible.

    Returns:
        The audio directory, the golden-set path, and a summary dictionary.
    """
    rows = read_metadata(dataset)
    source_audio = dataset / "audio"

    by_category: dict[str, list[str]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["filename"])

    rng = random.Random(seed)
    audio_dir = workdir / "audio"
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True)

    queries: list[dict[str, Any]] = []
    total_clips = 0
    for category in sorted(by_category):
        filenames = sorted(by_category[category])
        if per_class:
            filenames = rng.sample(filenames, min(per_class, len(filenames)))

        relevant: list[dict[str, Any]] = []
        for filename in filenames:
            shutil.copy2(source_audio / filename, audio_dir / filename)
            relevant.append(
                {
                    "filepath": str((audio_dir / filename).resolve()),
                    "start": 0.0,
                    "end": CLIP_SECONDS,
                }
            )
            total_clips += 1

        queries.append({"id": category, "query": phrase_for(category), "relevant": relevant})

    golden_path = workdir / "golden.json"
    golden_path.write_text(
        json.dumps({"name": "esc50", "queries": queries}, indent=2), encoding="utf-8"
    )

    summary = {
        "classes": len(queries),
        "clips": total_clips,
        "clips_per_class": len(queries[0]["relevant"]) if queries else 0,
    }
    return audio_dir, golden_path, summary


def run(args: argparse.Namespace) -> int:
    """Index the sampled corpus and evaluate retrieval on it.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"Building corpus from {args.dataset} ...")
    audio_dir, golden_path, summary = build_corpus(
        Path(args.dataset).resolve(), workdir, args.per_class, args.seed
    )
    print(
        f"  {summary['clips']} clips across {summary['classes']} classes "
        f"({summary['clips_per_class']} per class)"
    )

    store_dir = workdir / f"index-{args.backend}"
    if store_dir.exists():
        shutil.rmtree(store_dir)

    engine = EchoVector(
        store_dir=store_dir,
        backend=args.backend,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=0.0,
    )

    print(f"Indexing with backend={args.backend} ...")
    started = time.perf_counter()
    chunks = engine.index(audio_dir, batch_size=args.batch_size)
    index_seconds = time.perf_counter() - started
    print(
        f"  {chunks} chunks in {index_seconds:.1f}s "
        f"({chunks / max(index_seconds, 1e-9):.1f} chunks/sec)"
    )

    ks = tuple(int(part) for part in args.ks.split(",") if part.strip())
    print("Evaluating ...")
    report = evaluate(
        lambda query, k: engine.search(query, top_k=k, merge_spans=args.merge_spans),
        load_golden_set(golden_path),
        ks=ks,
    )

    print()
    print(report.format_table(include_per_query=args.per_query))

    if args.json_out:
        payload = report.to_dict()
        payload["run"] = {
            "backend": args.backend,
            "chunk_seconds": args.chunk_seconds,
            "merge_spans": args.merge_spans,
            "index_seconds": round(index_seconds, 2),
            "chunks": chunks,
            **summary,
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")

    return 0


def main() -> int:
    """Parse arguments and run the benchmark.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Extracted ESC-50 root directory.")
    parser.add_argument(
        "--workdir", default="./benchmark-esc50", help="Where to build the corpus."
    )
    parser.add_argument("--backend", default="clap", help="Embedding backend to evaluate.")
    parser.add_argument("--per-class", type=int, default=10, help="Clips per class; 0 for all 40.")
    parser.add_argument("--chunk-seconds", type=float, default=10.0, help="Chunk length.")
    parser.add_argument("--batch-size", type=int, default=16, help="Chunks embedded per batch.")
    parser.add_argument("--ks", default="1,5,10", help="Comma-separated cut-offs.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--per-query", action="store_true", help="Show a row per class.")
    parser.add_argument(
        "--no-merge-spans",
        dest="merge_spans",
        action="store_false",
        help="Score raw chunks instead of merged moments.",
    )
    parser.add_argument("--json-out", default="", help="Write the full report as JSON here.")
    parser.set_defaults(merge_spans=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
