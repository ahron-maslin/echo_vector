# 🔊 EchoVector

> **Semantic text search over audio files — without full transcription.**

[![CI](https://github.com/ahron-maslin/echo_vector/actions/workflows/workflow.yml/badge.svg)](https://github.com/ahron-maslin/echo_vector/actions/workflows/workflow.yml)
[![Coverage](https://img.shields.io/badge/coverage-%3E95%25-brightgreen)](.)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](.)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What is EchoVector?

EchoVector indexes audio by embedding the **waveform itself**, then finds moments in it from a
plain-language description of *how they sound* — no transcription anywhere in the pipeline.

```
Audio → Audio Chunks → Audio Embeddings ─┐
                                          ├─► ANN Search → Timestamped moments
Text Query → Text Embedding ──────────────┘
```

### Does it work?

Measured on [ESC-50](https://github.com/karoldvl/ESC-50) — 500 environmental recordings, 50
natural-language queries, 10 correct clips each, so chance is 0.020:

| | precision@1 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| **text → audio** (CLAP) | **0.900** | 0.800 | 0.937 | 0.835 |
| **audio → audio** (`similar`) | **0.950** | — | 0.966 | — |
| chance | 0.020 | 0.020 | — | — |

For 45 of 50 text queries the top result was already correct, and every query found a correct clip
in its top 10. Searching *with a clip* instead of a sentence does better still. Reproduce it with
`python benchmarks/esc50.py`; full methodology, per-class results, and known weaknesses are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

### What it is good at

The default backend, [CLAP](https://huggingface.co/laion/clap-htsat-unfused), is trained on audio
paired with captions that *describe sound* — "a dog barking", "rain on a metal roof", "a car engine
revving". So EchoVector is built for audio where the sound is the content:

- Sound libraries, foley, and sample packs
- Field recordings, wildlife, and bioacoustics
- Security and industrial monitoring
- Broadcast and archive material

### What it is not for

**Searching what people said.** CLAP encodes how audio *sounds*, not spoken content, so a query like
`"the part where we discussed pricing"` has no acoustic signature to match and will not work.

For speech, transcribe first and search the text — [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
runs faster than real time on a laptop and answers that question directly. EchoVector is a poor
substitute for transcription, and a good complement to it.

## Features

- 🎵 **Multi-format support** — MP3, WAV, FLAC, M4A, OGG, AIFF
- 🧠 **Direct audio embeddings** — No transcription needed
- 🔍 **Text → audio search** — Describe the sound you want
- 🪞 **Audio → audio search** — `similar` finds more that sounds like a clip you already have
- 🎯 **Moment-level results** — Consecutive hits fold into one span instead of five near-identical chunks
- 📐 **Measurable quality** — `eval` scores retrieval against a golden set (recall, nDCG, MRR, MAP)
- ⚡ **FAISS-powered** — Exact `flat`, or approximate `ivf` / `hnsw` for large archives
- 🔒 **Self-describing indexes** — a manifest pins the model that built a store, so it can't be queried with the wrong one
- ♻️ **Change-aware** — edited files are re-indexed, deleted ones are pruned, long runs checkpoint
- 🔌 **Pluggable backends** — CLAP for text search; wav2vec2, HuBERT, and AST for audio-to-audio
- 🧪 **Offline smoke backend** — `local` backend for CI/Kaggle tests without model downloads
- 🌐 **REST API** — Optional FastAPI server with background indexing jobs
- 📦 **Typed and tested** — strict mypy, ruff, >95% coverage

## Quick Start

### Installation

```bash
pip install "echo_vector[clap]"     # with semantic search (needs PyTorch)
pip install echo_vector             # ~50 MB, no model stack
```

Or with uv:

```bash
uv add "echo_vector[clap]"
```

**Why the extra?** PyTorch and transformers add roughly 4 GB — on Linux `pip` pulls the CUDA build
by default — and nothing except the model backends needs them. Indexing, search, the vector store,
and evaluation all run without. So the base install stays small and asks for the heavy half by name;
`echovector` tells you exactly what to run if you reach for a model backend without it.

On a machine with no GPU, install the much smaller CPU build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "echo_vector[clap]"     # ~200 MB total
```

The `clap` extra also enables the `wav2vec2`, `hubert`, and `ast` backends. The `local` backend
needs no extra, but it is a smoke-test stand-in rather than a real search backend.

### CLI Usage

```bash
# One-time indexing: split audio into timestamped chunks and embed each chunk.
# Directories are searched recursively; pass --no-recursive to stay at the top level.
echovector index ./sound-library

# Fast repeated search: embed only the text query and search the saved FAISS index
echovector search "glass breaking"

# Search with options
echovector search "heavy rain on a roof" --top-k 10 --min-score 0.2

# Already-indexed files are skipped; --force re-indexes them in place
echovector index ./sound-library --force

# View index statistics (reads the manifest — no model load)
echovector stats
```

For a no-download smoke test, use the deterministic local backend:

```bash
echovector index ./sound-library --backend local --store-dir ./ev-index
echovector search "high alarm tone" --backend local --store-dir ./ev-index
echovector stats --backend local --store-dir ./ev-index
```

### Finding more of the same sound

`similar` searches with a clip instead of a sentence. It never touches the text tower, so it also
works with the audio-only backends (`wav2vec2`, `hubert`, `ast`) that have no text side at all.

```bash
echovector similar ./samples/door-slam-03.wav --top-k 10
```

### Hearing what you found

A result is a file plus a time range, so the next step is usually to listen to it:

```bash
# Write the moment out as its own file, with a little context either side
echovector extract ./library/street.wav 124.0 134.0 --out ./clip.wav --padding 1.5

# Or print a player command to run yourself (nothing is executed for you)
echovector play ./library/street.wav 124.0 134.0 --player mpv
```

### Keeping an index honest

Files are tracked by content signature, not just path, so an edited recording is re-read on the next
index run rather than silently going stale.

```bash
# Report changed files and drop ones deleted from disk
echovector prune

# See what would change without touching the index
echovector prune --check
```

### Measuring search quality

Retrieval quality is the thing worth optimising, so it is worth measuring rather than guessing.
Describe some queries and the moments that should answer them, then score the index:

```json
{
  "name": "library-smoke",
  "queries": [
    {
      "query": "glass breaking",
      "relevant": [{"filepath": "/audio/foley-03.wav", "start": 12.0, "end": 20.0}]
    }
  ]
}
```

```bash
echovector eval ./golden.json --ks 1,5,10 --per-query
```

```
metric         k=1      k=5     k=10
------------------------------------
recall       0.389    1.000    1.000
precision    1.000    0.600    0.333
hit_rate     1.000    1.000    1.000
mrr          1.000    1.000    1.000
ndcg         1.000    1.000    1.000
map          1.000    1.000    1.000

Span fit (mean IoU with the annotated span): 0.519
Latency: mean 7.7 ms | p50 5.6 ms | p95 16.6 ms | 130.5 queries/sec
```

A result counts as relevant when it overlaps an annotated span, so chunk boundaries do not have to
line up with your annotations. **Span fit** is separate and worth watching: the ranking metrics say
whether the right moment was found, while span fit says how tightly the result bounds it — a result
that covers the event but runs three times too long scores perfectly on the former and poorly here. Use `eval` to compare backends, chunk lengths, or index types against
each other instead of by feel.

> The figures above come from a synthetic tone corpus used to exercise the harness. For real
> measured quality on a public dataset, see [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

The search command does not reopen or scan the audio files. All expensive audio processing happens
during `index`; `search` loads the saved vector index, embeds the short text query, and returns the
nearest timestamped chunks.

### Python API

```python
from echovector import EchoVector

ev = EchoVector()

# Index audio files
ev.index("./sound-library")

# Search with natural language
results = ev.search("glass breaking")

for r in results:
    print(
        f"{r.filepath} "
        f"[{r.timestamp_range.start:.1f}s - {r.timestamp_range.end:.1f}s] "
        f"score={r.score:.4f}"
    )
```

### Moments, not chunks

Indexing slices audio on a fixed grid, so one long relevant passage would otherwise come back as
several near-identical hits that crowd out every other file. `search` folds consecutive hits from
the same file into a single span:

```python
ev.search("rain on a roof", top_k=5)                     # moments (default)
ev.search("rain on a roof", top_k=5, merge_spans=False)  # raw chunks
ev.search("rain on a roof", top_k=5, max_per_file=1)     # one hit per file
```

A neighbouring chunk is only absorbed if it scores within 90% of the file's best hit, so a strong
match cannot widen itself over the unrelated audio next to it. Each merged result records
`merged_chunk_count` and `chunk_ids` in its metadata.

### Finding similar audio

```python
results = ev.search_by_audio("./samples/door-slam-03.wav", top_k=10)
```

Takes a path or a mono waveform, skips the text tower entirely, and excludes the query file from its
own results by default (`exclude_source=False` to keep it).

### Filtering results

```python
from echovector import EchoVector, SearchFilter

ev = EchoVector()
results = ev.search(
    "glass breaking",
    top_k=10,
    filters=SearchFilter(min_score=0.2, filepaths=["/audio/foley-03.wav"]),
)
```

Filters run after retrieval, so EchoVector over-fetches candidates to still
return a full page of `top_k` results.

### Keeping the index current

```python
ev.is_stale("./library/street.wav")  # changed since it was indexed?
ev.stale_files()                     # everything needing a re-index
ev.prune()                           # drop files deleted from disk
```

Long indexing runs checkpoint the vector index periodically
(`index(..., checkpoint_every_chunks=2000)`), so an interruption keeps its progress instead of
leaving metadata rows with no matching vectors.

### Large archives

`flat` is exact and the right default for a personal library. For bigger collections, trade a little
recall for speed:

```python
ev = EchoVector(index_type="hnsw", index_params={"ef_search": 128})
```

The structure is recorded in the manifest, so reopening a store uses what is actually on disk and
warns if you asked for something else. Use `echovector eval` to measure the recall you are trading
away rather than assuming it.

### Chunk length

The default is **5 seconds**, chosen by measurement rather than assumption: on a
long-form benchmark, quality falls away monotonically as chunks grow past that,
because a long chunk blends several unrelated sounds into one embedding. At the
old 10s default, nDCG@10 was 0.738 against 0.882 at 5s. Anything in roughly the
3-5s band works well; see [benchmarks/RESULTS.md](benchmarks/RESULTS.md) for the
full sweep, and use `echovector eval` to tune it for your own audio.

`chunk_seconds` must also fit the window the backend can embed. CLAP crops anything
longer than 10s to a *random* 10s window, which would make the embedding both
non-deterministic and unrepresentative of the timestamp range stored beside it.
EchoVector rejects the combination rather than indexing something misleading:

```python
EchoVector(chunk_seconds=30.0)  # with CLAP: ValueError at index time
```

## Testing on Kaggle

Kaggle is useful for GPU-backed CLAP tests, but first check the runtime Python version:

```python
import sys
print(sys.version)
```

EchoVector currently declares `Python >=3.12`. If the Kaggle image is older, install and test in a
Python 3.12-capable environment instead, or relax the project requirement only after validating the
test suite on that Python version.

### Notebook smoke test without internet/model downloads

Upload this repository as a Kaggle dataset, attach it to a notebook, then run:

```python
%cd /kaggle/input/<your-echo-vector-dataset>
!pip install -e . --no-deps
!pip install numpy soundfile librosa faiss-cpu typer rich pydantic
!python -m pytest tests/ -q
```

Create a tiny audio corpus and test the real CLI/index path:

```python
import os
import numpy as np
import soundfile as sf

audio_dir = "/kaggle/working/ev-audio"
index_dir = "/kaggle/working/ev-index"
os.makedirs(audio_dir, exist_ok=True)

sr = 16000
t = np.linspace(0, 1.0, sr, endpoint=False)
sf.write(f"{audio_dir}/high_tone.wav", 0.25 * np.sin(2 * np.pi * 880 * t), sr)
sf.write(f"{audio_dir}/low_tone.wav", 0.25 * np.sin(2 * np.pi * 110 * t), sr)
```

```python
!echovector index /kaggle/working/ev-audio --backend local --store-dir /kaggle/working/ev-index --reset
!echovector search "high alarm tone" --backend local --store-dir /kaggle/working/ev-index --top-k 2
!echovector stats --backend local --store-dir /kaggle/working/ev-index
```

This validates packaging, audio loading, FAISS persistence, metadata storage, and the CLI without
depending on Hugging Face downloads.

### CLAP semantic test

For actual semantic text-to-audio search, enable internet in the notebook settings and use a GPU
runtime if available:

```python
!pip install transformers torch faiss-cpu librosa soundfile
!echovector index /kaggle/input/<audio-dataset> --backend clap --device cuda --store-dir /kaggle/working/clap-index --reset
!echovector search "a dog barking in the distance" --backend clap --device cuda --store-dir /kaggle/working/clap-index --top-k 10
```

If GPU is unavailable, replace `--device cuda` with `--device cpu`; it will be slower. Keep indexes
under `/kaggle/working` so they are writable during the notebook session.

## Architecture

```
echovector/
├── audio/        # Loading, chunking, clip extraction, metadata
├── embeddings/   # Pluggable embedding backends (CLAP, wav2vec2, HuBERT, AST, local)
├── indexing/     # FAISS index, SQLite metadata store, index manifest
├── search/       # Result models, filtering, span merging
├── evaluation/   # Retrieval metrics, golden sets, evaluation harness
├── cli/          # Typer-based CLI with Rich output
├── api/          # Optional FastAPI server
└── utils/        # Logging helpers
```

### How a store is laid out

```
<store-dir>/
├── index.faiss       # the vectors (FAISS IndexIDMap2 over flat / IVF / HNSW)
├── metadata.sqlite   # chunk -> source file, timestamps, file signature
└── manifest.json     # backend, dimension, chunking, and index structure
```

The manifest is what makes a store safe to reopen. Vectors carry no record of
the model that produced them, so without it an index built with CLAP could be
queried with another backend of the same width and would return confident,
meaningless scores. EchoVector compares the manifest on open and refuses the
mismatch instead.

## Supported Embedding Backends

| Backend | Text → audio | Audio → audio | Needs `[clap]` | Notes |
|---------|--------------|---------------|----------------|-------|
| **CLAP** (default) | ✅ | ✅ | yes | Text and audio share one space; the only backend that answers text queries |
| `local` | ✅ | ✅ | no | Deterministic acoustic features. For offline smoke tests, not real search |
| `wav2vec2` | ❌ | ✅ | yes | Speech-pretrained encoder, mean-pooled |
| `hubert` | ❌ | ✅ | yes | Speech-pretrained encoder, mean-pooled |
| `ast` | ❌ | ✅ | yes | Audio Spectrogram Transformer, AudioSet-pretrained |

The audio-only backends have no text tower, so a text query against them raises
`TextSearchUnsupportedError` pointing you at `search_by_audio`. Use them with `similar` when you
care about acoustic resemblance rather than description.

## Development

```bash
# Clone and install
git clone https://github.com/ahron-maslin/echo_vector.git
cd echo_vector
uv sync --all-extras

# Run checks
make lint
make typecheck
make test
make coverage
```

## REST API

```bash
pip install "echo_vector[api]"
```

```python
import uvicorn
from echovector import EchoVector
from echovector.api.server import app, configure_engine

configure_engine(EchoVector(store_dir="./ev-index", backend="local"))
uvicorn.run(app, host="127.0.0.1", port=8000)
```

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness; needs no configured engine |
| `POST` | `/index` | Starts a background job, returns `202` with a job id |
| `GET` | `/jobs` · `/jobs/{id}` | Job status, progress, and errors |
| `POST` | `/search` | Text query against the index |
| `GET` | `/stats` | Index statistics |
| `POST` | `/reset` | Clears the index |

Indexing runs on a background worker because a real corpus takes minutes to hours — a synchronous
call would simply time out. One job runs at a time; a second request gets `409` naming the active
job. Reads stay available throughout, since the worker takes the engine lock per file rather than
for the whole run. Jobs live in memory and do not survive a restart.

### Securing it

Set `ECHOVECTOR_API_TOKEN` to require `Authorization: Bearer <token>` on `/index`, `/reset`, and the
job routes. Leave it unset and those routes stay open, which is the local-use default.

A token controls *who* can call those routes; it does not make them safe to expose publicly.
`POST /index` still reads whatever filesystem paths the request names, so anyone holding the token
can make the server read any file it can read, and `POST /reset` deletes the index irreversibly.
Bind to localhost, or put it behind an authenticating proxy on a host whose filesystem you are
willing to expose.

## License

MIT — see [LICENSE](LICENSE).
