# Retrieval benchmark results

The first measured answer to the question the project had never asked: **does
text-to-audio search actually work?**

Reproduce with:

```bash
# Text -> audio retrieval, and the no-semantics floor to read it against
python benchmarks/esc50.py --dataset ./ESC-50-master --backend clap --per-class 10
python benchmarks/esc50.py --dataset ./ESC-50-master --backend local --per-class 10

# Chunk length and span merging, on concatenated long-form recordings
python benchmarks/esc50_longform.py --dataset ./ESC-50-master --chunk-seconds 2,3,4,5,6,7,8,10

# Audio -> audio ("find more that sounds like this")
python benchmarks/esc50_similar.py --dataset ./ESC-50-master --backend clap
```

All figures below: CPU-only 8-core i7-1165G7, `torch 2.12`, `transformers 4.57`,
`faiss 1.14`, `laion/clap-htsat-unfused`, flat index.

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
| Index throughput | 4.1 chunks/sec | 13.4 chunks/sec |
| 500 clips indexed in | 122s | 37s |
| Query latency (p50) | 67 ms | 1.6 ms |

Query latency is dominated by embedding the query text through CLAP, not by the
vector search. At this rate a 60,000-file library takes roughly four hours to
index on CPU; a GPU changes that substantially.

The `local` backend was 29.7 chunks/sec when its spectral features came from
librosa, whose numba-compiled kernels are faster than the plain numpy that
replaced them. That trade bought a 340 MB smaller install and costs nothing that
matters: `local` exists for smoke tests, and CLAP — the backend anyone measures
with — is unaffected, since resampling moved to soxr, which is what librosa used
underneath. Its retrieval scores are unchanged: still exactly at chance.

## ESC-50 long-form — chunk length and span merging

ESC-50 clips are uniform 5s single-source recordings, so every clip is exactly
one chunk: chunk length never varies and span merging never fires. To evaluate
either, `benchmarks/esc50_longform.py` concatenates clips into 60-second
recordings. Placement is ours, so the ground truth is exact to the sample, and
shuffling means each file holds unrelated sounds back to back — which is what
makes chunk misalignment bite.

**25 recordings of 60s, 300 clips placed, 50 queries.**

| chunk | merge | recall@10 | nDCG@10 | precision@10 | MRR |
|-------|-------|-----------|---------|--------------|-----|
| 2s | off | 0.730 | 0.719 | 0.736 | 0.895 |
| 2s | **on** | **0.800** | **0.790** | 0.664 | 0.897 |
| 5s | off | 0.907 | **0.882** | 0.544 | 0.912 |
| 5s | on | 0.900 | 0.877 | 0.540 | 0.912 |
| 10s | off | 0.720 | 0.738 | 0.432 | 0.926 |
| 10s | on | 0.727 | 0.742 | 0.436 | 0.926 |

### Chunk length: the 10s default was costing real quality

Swept across eight lengths, merging off so the comparison is unaffected by it:

| chunk | 2s | 3s | 4s | **5s** | 6s | 7s | 8s | 10s |
|-------|----|----|----|--------|----|----|----|-----|
| nDCG@10 | 0.719 | 0.833 | 0.807 | **0.882** | 0.777 | 0.744 | 0.701 | 0.738 |
| recall@10 | 0.730 | 0.840 | 0.803 | 0.907 | 0.767 | 0.760 | 0.707 | 0.720 |

The shape is unambiguous above 5s: quality falls away monotonically as chunks
grow, because a long chunk blends several unrelated sounds into one embedding.
At 10s — twice the event length here — nDCG@10 is 0.738 against 0.882 at 5s.

**The 5s peak is partly an artefact and should not be over-read.** 5s aligns
exactly with the synthetic clip boundaries, and it is a visible outlier against
its neighbours (0.807 at 4s, 0.777 at 6s). Among lengths that align with
*nothing*, **3s scores highest** at 0.833, and interpolating its neighbours puts
an unaligned 5s nearer 0.79.

So what the sweep robustly establishes is a **band, not a point**: somewhere
around 3–5s beats 7–10s substantially, and the old 10s default sat well outside
it. The default is now **5 seconds** — inside the good band, and cheaper than 3s,
which produces 67% more vectors for a gain this corpus cannot cleanly confirm.
Users whose audio has shorter events should try 3s and measure with `eval`.

Going below 3s hurts: 2s fragments lose the context CLAP needs, and it
repeat-pads them to fill its 10s window regardless.

### Span merging: the relevance band earns its place, and it is chunk-dependent

Merging is gated by a relevance band (`core.SPAN_MERGE_RELATIVE_SCORE`, default
0.9): a neighbouring chunk joins a span only if it scores within that fraction of
the file's best hit. Sweeping it looks, at first, like an argument for removing
it — at 2s chunks a band of 0.0 scores recall@10 0.870 / nDCG@10 0.838 against
0.800 / 0.790 at 0.9.

Ranking metrics alone cannot settle this, because none of them ask whether a
returned span actually *fits* the event it found. Adding temporal IoU against the
annotated span does, and it changes the answer. (IoU was computed ad hoc for this
analysis and, having proved decisive, is now a first-class metric: every
`echovector eval` run reports "Span fit".)

**2s chunks** — chunks shorter than the 5s events

| band | mean span | max span | **IoU** | recall@10 | nDCG@10 |
|------|-----------|----------|---------|-----------|---------|
| 0.00 | 4.34s | 10.0s | **0.383** | 0.870 | 0.838 |
| 0.80 | 3.16s | 8.0s | 0.351 | 0.850 | 0.822 |
| 0.90 | 2.66s | 8.0s | 0.324 | 0.800 | 0.790 |
| 0.95 | 2.32s | 6.0s | 0.300 | 0.767 | 0.761 |

**5s chunks** — chunks matching the 5s events

| band | mean span | max span | **IoU** | 
|------|-----------|----------|---------|
| 0.00 | 5.48s | **20.0s** | 0.503 |
| 0.80 | 4.92s | 15.0s | 0.535 |
| 0.90 | 4.85s | 10.0s | **0.536** |
| 0.95 | 4.83s | 5.0s | **0.544** |

The two regimes want opposite things, and the reason is structural:

- **When chunks are shorter than events**, adjacent hits are usually *the same
  event fragmented*, so merging aggressively reassembles it. A low band wins on
  every metric including IoU (0.383 vs 0.324), and spans stay bounded.
- **When chunks match event length**, adjacent hits are usually *different
  events*, so merging them is a mistake. A low band chains up to 20s — four
  unrelated events in one span — and IoU degrades (0.503 vs 0.544).

At the new 5s default the band is doing exactly the job it was added for, and
0.9 sits within noise of the best value measured (0.536 vs 0.544 at 0.95). It
stays at 0.9.

Two things worth recording honestly. First, the reasoning originally given for
the band — "a strong hit should not absorb the unrelated audio next to it" —
turns out to be right only in the chunks-match-events regime; in the other
regime it actively hurts. Second, the right long-term fix is probably not a
constant at all but a band derived from the ratio of chunk length to observed
event length.

## ESC-50 — audio → audio retrieval

`benchmarks/esc50_similar.py` uses each clip in turn as the query and counts a
result correct when it comes from the same class. The query clip is excluded, so
9 of 500 remain correct and chance is **0.018**.

| Metric | `similar` (audio→audio) | text→audio | chance |
|--------|------------------------|------------|--------|
| precision@1 | **0.950** | 0.900 | 0.018 |
| precision@5 | **0.912** | 0.904 | 0.018 |
| precision@10 | 0.765 | 0.800 | 0.018 |
| MRR | **0.966** | 0.937 | — |

**Similarity search is the stronger capability.** It beats text search on
precision@1 (0.950 vs 0.900) and MRR (0.966 vs 0.937), which makes sense: it
compares audio to audio and never has to cross the modality gap that text search
depends on. It is also the only mode available to the `wav2vec2`, `hubert`, and
`ast` backends, which have no text tower at all.

The cost is query latency: **614 ms** against 67 ms for a text query, because the
query clip must be decoded and embedded rather than just a short string.

## Approximate indexes

`flat` is exhaustive and exact. `ivf` and `hnsw` trade recall for speed. Measured
against the flat baseline on the same 500 real CLAP embeddings, using the 50
ESC-50 text queries:

| structure | settings | recall@10 vs flat |
|-----------|----------|-------------------|
| `hnsw` | defaults (`ef_search=64`) | **1.000** |
| `hnsw` | `ef_search=16` | 0.994 |
| `ivf` | `nlist=20` (default `nprobe`) | **1.000** |
| `ivf` | `nlist=20, nprobe=4` | 0.966 |

At the shipped defaults both are effectively lossless here, and degrade
gracefully when tuned down. **This is a floor, not a promise:** 500 vectors is
far below the scale where an approximate index is worth using at all, and recall
generally falls as a corpus grows. Treat these as evidence the defaults are not
broken, and measure your own corpus with `eval` before trusting them on it.

One caveat found while measuring: a corpus of near-identical vectors — pure tones
a few Hz apart, where the top-15 scores span 0.008 — gives 0.00 overlap between
flat and ANN results. Nothing is wrong; when hundreds of candidates tie to four
decimal places, which ten come back is arbitrary in any structure. It is a
reminder that ANN recall is only meaningful on a corpus with real spread.

## What this does and does not establish

**Does:** EchoVector retrieves environmental and non-speech audio from a plain
description, well above chance and in line with the published capability of the
underlying model.

**Does not:** say anything about speech content. CLAP is trained on captions
describing sound, so it cannot answer "where did they discuss pricing" — see the
README's scope section. ESC-50 clips are also uniform, clean, five-second, and
single-source; a real library of long, noisy, overlapping recordings is harder.

## Still to measure

- **Chunk length on natural long-form audio.** Measured here on concatenated
  clips, where event boundaries are uniform and known. A corpus of genuinely
  continuous recordings (Clotho, AudioCaps) would confirm the shape without the
  synthetic regularity.
- **An adaptive merge band.** The optimal value depends on chunk length relative
  to event length, which a constant cannot express.
- **Approximate indexes at scale.** Measured above at 500 vectors, which is far
  below the size where they earn their keep. Recall generally falls as a corpus
  grows, so the interesting measurement is at 10^5-10^6 chunks.
