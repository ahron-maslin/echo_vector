# 🔊 EchoVector

> **Semantic text search over audio files — without full transcription.**

[![CI](https://github.com/echovector/echovector/actions/workflows/test.yml/badge.svg)](https://github.com/echovector/echovector/actions/workflows/test.yml)
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
- 🔌 **Pluggable backends** — CLAP, Whisper, wav2vec2, HuBERT, AST
- 📊 **Rich CLI** — Progress bars, colors, benchmarking mode
- 🌐 **REST API** — Optional FastAPI server
- 📦 **Production-ready** — Typed, tested, documented

## Quick Start

### Installation

```bash
pip install echovector
```

Or with uv:

```bash
uv add echovector
```

### CLI Usage

```bash
# Index a directory of audio files
echovector index ./meetings

# Search across indexed audio
echovector search "discussion about transformers"

# Search with options
echovector search "pricing strategy" --top-k 10

# View index statistics
echovector stats
```

### Python API

```python
from echovector import EchoVector

ev = EchoVector()

# Index audio files
ev.index("./meetings")

# Search with natural language
results = ev.search("conversation about CUDA kernels")

for r in results:
    print(f"{r.file} [{r.start:.1f}s - {r.end:.1f}s] score={r.score:.4f}")
```

## Architecture

```
echovector/
├── audio/        # Audio loading, chunking, streaming, metadata
├── embeddings/   # Pluggable embedding backends (CLAP, Whisper, etc.)
├── indexing/     # Vector index backends (FAISS, with pluggable design)
├── search/       # Search engine, filtering, result hydration
├── cli/          # Typer-based CLI with Rich output
├── api/          # Optional FastAPI server
├── evaluation/   # Metrics (recall@k, throughput)
├── benchmarks/   # Reproducible benchmark harness
└── utils/        # Config, logging, helpers
```

## Supported Embedding Backends

| Backend | Text+Audio Aligned | Notes |
|---------|-------------------|-------|
| **CLAP** (default) | ✅ | Best for text→audio search |
| Whisper Encoder | ❌ | Audio-only embeddings |
| wav2vec2 | ❌ | Audio-only, good for speech |
| HuBERT | ❌ | Audio-only, self-supervised |
| Audio Spectrogram Transformer | ❌ | Audio-only, classification-focused |

## Development

```bash
# Clone and install
git clone https://github.com/echovector/echovector.git
cd echovector
uv sync --all-extras

# Run checks
make lint
make typecheck
make test
make coverage
```

## License

MIT
