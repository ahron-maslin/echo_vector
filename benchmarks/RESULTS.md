# Retrieval benchmark results

The first measured answer to the question the project had never asked: **does
text-to-audio search actually work?**

Reproduce with:

```bash
python benchmarks/esc50.py --dataset ./ESC-50-master --backend clap --per-class 10
python benchmarks/esc50.py --dataset ./ESC-50-master --backend local --per-class 10
```

## ESC-50 — text → audio retrieval

[ESC-50](https://github.com/karoldvl/ESC-50) is 2000 five-second environmental
recordings across 50 balanced classes. Each class name becomes a natural-language
query ("a baby crying", "glass breaking", "a chainsaw"), and the 10 sampled clips
of that class are its relevant results. Corpus: **500 clips, 50 queries, 10
relevant each**, so random guessing scores 10/500 = **0.020**.

| Metric | CLAP | `local` (floor) | Chance |
|--------|------|-----------------|--------|
| precision@1 | **0.900** | 0.020 | 0.020 |
| precision@10 | **0.800** | 0.020 | 0.020 |
| recall@10 | **0.800** | 0.020 | 0.020 |
| hit_rate@10 | **1.000** | 0.120 | — |
| MRR | **0.937** | 0.040 | — |
| nDCG@10 | **0.835** | 0.020 | — |
| MAP@10 | **0.760** | 0.009 | — |

**Read this as:** for 45 of 50 queries the single top result was already correct.
Every query found at least one correct clip in its top 10. On average 8 of the 10
correct clips were in the top 10. CLAP scores **45x above chance** at
precision@1, and the `local` backend sits exactly at chance — which is what it
should do, and confirms the harness is measuring retrieval rather than an
artefact of the corpus.

CLAP's published zero-shot ESC-50 accuracy is roughly 82–91%. Landing at 90%
precision@1 is an independent check that the whole pipeline — chunking,
normalization, FAISS, and scoring — is wired up correctly end to end.

### Where it struggles

| Class | recall@10 | Likely cause |
|-------|-----------|--------------|
| insects | 0.100 | Confused with `crickets` (0.900), which is a kind of insect |
| wind | 0.300 | Overlaps `rain`, `sea_waves`, `thunderstorm` |
| cow | 0.400 | Generic livestock confusion with `sheep`, `pig`, `hen` |
| engine | 0.500 | `helicopter`, `airplane`, `train`, `vacuum_cleaner` all contain engines |

These are semantic overlaps in the label set rather than retrieval failures — the
"wrong" answers are usually a genuinely similar sound. It does show that query
phrasing carries real weight: distinguishing overlapping categories needs a more
specific description than a bare class name.

### Cost

| | CLAP | `local` |
|---|---|---|
| Index throughput | 4.1 chunks/sec | 29.7 chunks/sec |
| 500 clips indexed in | 122s | 17s |
| Query latency (p50) | 67 ms | 1.6 ms |

CPU only (8-core i7-1165G7), `torch 2.12`, `transformers 4.57`, `faiss 1.14`,
flat index. Query latency is dominated by embedding the query text through CLAP,
not by the vector search. At this rate a 60,000-file library takes roughly four
hours to index on CPU; a GPU changes that substantially.

## What this does and does not establish

**Does:** EchoVector retrieves environmental and non-speech audio from a plain
description, well above chance and in line with the published capability of the
underlying model.

**Does not:** say anything about speech content. CLAP is trained on captions
describing sound, so it cannot answer "where did they discuss pricing" — see the
README's scope section. ESC-50 clips are also uniform, clean, five-second, and
single-source; a real library of long, noisy, overlapping recordings is harder.

## Still to measure

- **Chunk length.** ESC-50 clips are all 5s and fit CLAP's 10s window whole, so
  chunking never varies here. Needs a long-form corpus (Clotho, AudioCaps) to
  test properly.
- **Span merging.** Also inert on single-chunk clips. The 0.9 relevance band in
  `core.SPAN_MERGE_RELATIVE_SCORE` was set by reasoning and is still unvalidated.
- **Approximate indexes.** `ivf` and `hnsw` recall against the `flat` baseline at
  a corpus size where they matter.
- **Audio-to-audio.** `search_by_audio` has no benchmark at all yet.
