# 🔊 EchoVector

> **Semantic text search over audio files — without full transcription.**

[![CI](https://github.com/ahron-maslin/echo_vector/actions/workflows/workflow.yml/badge.svg)](https://github.com/ahron-maslin/echo_vector/actions/workflows/workflow.yml)
[![Coverage](https://img.shields.io/badge/coverage-%3E95%25-brightgreen)](.)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](.)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What is EchoVector?

EchoVector indexes audio files by generating **semantic embeddings directly from audio waveforms**, then lets you search them with natural language text queries — all without transcribing a single word.

### Traditional approach (slow & expensive)

```
Audio → Full Transcription → Text Embeddings → Text Search
```

### EchoVector approach (fast & efficient)

```
Audio → Audio Chunks → Audio Embeddings ─┐
                                          ├─► ANN Search → Results
Text Query → Text Embedding ──────────────┘
```

## Features

- 🎵 **Multi-format support** — MP3, WAV, FLAC, M4A
- 🧠 **Direct audio embeddings** — No transcription needed
- 🔍 **Semantic search** — Query with natural language
- ⚡ **FAISS-powered** — Approximate nearest neighbor search
- 🔌 **Pluggable backends** — CLAP implemented today; Whisper, wav2vec2, HuBERT, AST are stubbed for future work
- 🧪 **Offline smoke backend** — `local` backend for CI/Kaggle tests without model downloads
- 🔒 **Self-describing indexes** — a manifest pins the model that built a store, so it can't be queried with the wrong one
- 📊 **Rich CLI** — Progress bars, colors, result filtering
- 🌐 **REST API** — Optional FastAPI server
- 📦 **Typed and tested** — strict mypy, ruff, >95% coverage

## Quick Start

### Installation

```bash
pip install echo_vector
```

Or with uv:

```bash
uv add echo_vector
```

### CLI Usage

```bash
# One-time indexing: split audio into timestamped chunks and embed each chunk.
# Directories are searched recursively; pass --no-recursive to stay at the top level.
echovector index ./meetings

# Fast repeated search: embed only the text query and search the saved FAISS index
echovector search "discussion about transformers"

# Search with options
echovector search "pricing strategy" --top-k 10 --min-score 0.2

# Already-indexed files are skipped; --force re-indexes them in place
echovector index ./meetings --force

# View index statistics (reads the manifest — no model load)
echovector stats
```

For a no-download smoke test, use the deterministic local backend:

```bash
echovector index ./meetings --backend local --store-dir ./ev-index
echovector search "high alarm tone" --backend local --store-dir ./ev-index
echovector stats --backend local --store-dir ./ev-index
```

The search command does not reopen or scan the audio files. All expensive audio processing happens
during `index`; `search` loads the saved vector index, embeds the short text query, and returns the
nearest timestamped chunks.

### Python API

```python
from echovector import EchoVector

ev = EchoVector()

# Index audio files
ev.index("./meetings")

# Search with natural language
results = ev.search("conversation about CUDA kernels")

for r in results:
    print(
        f"{r.filepath} "
        f"[{r.timestamp_range.start:.1f}s - {r.timestamp_range.end:.1f}s] "
        f"score={r.score:.4f}"
    )
```

### Filtering results

```python
from echovector import EchoVector, SearchFilter

ev = EchoVector()
results = ev.search(
    "conversation about CUDA kernels",
    top_k=10,
    filters=SearchFilter(min_score=0.2, filepaths=["/audio/standup.wav"]),
)
```

Filters run after retrieval, so EchoVector over-fetches candidates to still
return a full page of `top_k` results.

### Chunk length and the model's window

`chunk_seconds` must fit the window the backend can embed. CLAP crops anything
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
!echovector index /kaggle/input/<audio-dataset> --backend clap --device cuda --store-dir /kaggle/working/clap-index --recursive --reset
!echovector search "people discussing pricing strategy" --backend clap --device cuda --store-dir /kaggle/working/clap-index --top-k 10
```

If GPU is unavailable, replace `--device cuda` with `--device cpu`; it will be slower. Keep indexes
under `/kaggle/working` so they are writable during the notebook session.

## Architecture

```
echovector/
├── audio/        # Audio loading, chunking, streaming, metadata
├── embeddings/   # Pluggable embedding backends (CLAP, Whisper, etc.)
├── indexing/     # Vector index backends (FAISS, with pluggable design)
├── search/       # Search engine, filtering, result hydration
├── cli/          # Typer-based CLI with Rich output
├── api/          # Optional FastAPI server
├── evaluation/   # Vector similarity metrics
└── utils/        # Logging helpers
```

### How a store is laid out

```
<store-dir>/
├── index.faiss       # the vectors (FAISS IndexIDMap2 over IndexFlatIP)
├── metadata.sqlite   # chunk -> source file, timestamps, audio metadata
└── manifest.json     # which backend, dimension, and chunking built the index
```

The manifest is what makes a store safe to reopen. Vectors carry no record of
the model that produced them, so without it an index built with CLAP could be
queried with another backend of the same width and would return confident,
meaningless scores. EchoVector compares the manifest on open and refuses the
mismatch instead.

## Supported Embedding Backends

| Backend | Status | Notes |
|---------|--------|-------|
| **CLAP** (default) | ✅ Implemented | Text+audio aligned; best for text→audio search |
| `local` | ✅ Implemented | Deterministic acoustic-feature backend for offline smoke tests |
| Whisper Encoder | 🚧 Stub | Raises `NotImplementedError`; not yet implemented |
| wav2vec2 | 🚧 Stub | Raises `NotImplementedError`; not yet implemented |
| HuBERT | 🚧 Stub | Raises `NotImplementedError`; not yet implemented |
| Audio Spectrogram Transformer | 🚧 Stub | Raises `NotImplementedError`; not yet implemented |

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

The optional FastAPI server has **no authentication**: `POST /index` reads any
filesystem path the request names and `POST /reset` deletes the index. Run it on
localhost, or behind a proxy that authenticates those routes — never bound to a
public interface as-is.

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

## License

MIT — see [LICENSE](LICENSE).
