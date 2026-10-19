# Goals, and where help is wanted

## The goal

Find the part of a recording where a topic is discussed, from a plain-language query, across
hundreds of hours of spoken audio. The motivating corpus is about 237 recorded shiurim (roughly
60 audio-hours), with Hebrew and English mixed in the same sentence.

The idea this project started from:

> Turn the query text into an audio-side representation (a "generated audio fingerprint"), then
> match that against the audio directly, with no transcription step.

The appeal is real: transcription is the slow part, and if the query could be matched against
audio features directly, indexing would be cheap. We want this to work. The sections below say
what has been measured so far, so you can start from there instead of repeating it.

## What has been tried

| Approach | Result |
|---|---|
| CLAP (text to waveform embedding) | Excellent for sounds (precision@1 0.90 on ESC-50). Cannot match speech content: it encodes how audio sounds, not what is said. |
| Query text to phonemes (espeak G2P), then a CTC keyword scan over wav2vec2 phoneme posteriorgrams | Works for exact words, but only 1.6x faster than whisper-small (8.5x vs 5.4x realtime), because the acoustic encoder dominates the cost of both. |
| Same, as a search system | No index, so every query rescans every posteriorgram: about 4.6 minutes per query over 60 hours. Posteriorgrams take about 282 MB per audio-hour (about 17 GB total). Verbatim only, so no concept reach. |
| Transcribe once, embed the text, expand queries by sound (what ships today) | Works. Transcription is a one-time cost (about 11 hours for the corpus). Concept queries and misspelled Hebrew terms both land on the right passages in the one real transcript tested. Not yet evaluated on the full corpus. |

The spike code for the acoustic experiments lives in `spikes/` (not part of the package).

## Why "match text against audio features" is hard for speech

- Concept search needs linguistic content. A query like "the threshold at the entrance" shares
  no acoustic pattern with a passage that says "escoopa". Something has to recover meaning from
  speech, and doing it with an audio encoder costs as much as transcribing.
- Direct matching needs a joint space where a text query and a stretch of speech land near each
  other. For sound events that exists (CLAP). For spoken content it is an open research
  problem, not an off-the-shelf model.
- Whisper spells the same Hebrew term several ways within one file ("escoopa" 33 times,
  "scuba" 8, "is scuba" 4), so text-side search needs help even after transcription.

## Open directions worth trying

These are untested hypotheses, not claims.

1. **A joint text-speech embedding for spoken content.** Train or find a model that maps a text
   query and a window of speech into one space, so search is a vector lookup with no
   transcript. Candidates: a contrastive projection on top of frozen whisper-encoder or
   HuBERT features, using the transcripts we already have as paired training data.
2. **Make acoustic keyword search indexable.** The scan failed on query cost, not accuracy.
   Phone n-gram or lattice indexes (an inverted index over recognised phoneme sequences) could
   turn the 4.6 minutes per query into a lookup. Storage and recall are the open questions.
3. **Cheaper transcription.** If the encoder is the bottleneck, test distilled or smaller
   multilingual encoders against whisper-small on Hebrew/English code-switching.
4. **Better handling of Hebrew terms.** A shareable alias map for common terms, and
   transliteration-aware matching, would improve the shipping text path regardless of 1-3.
5. **Evaluation.** There is no labelled set of "this query should land on this passage" for the
   shiur corpus. A small golden set would let every idea above be compared fairly. The
   evaluation harness in `echovector/evaluation/` already supports golden sets.

## How to judge a new approach

Against the shipping path (transcribe, embed text, phonetic expansion), a new approach should
beat it on at least one of:

- indexing cost per audio-hour (transcription today: about 11 hours for 60 hours of audio),
- query latency over the whole corpus (today: effectively instant),
- retrieval quality on concept queries and Hebrew terms, measured on a golden set.

If it does not beat it on any of these, it is not worth the added complexity. Negative results
are useful too; please add them to the table above with numbers.

## Getting started

```bash
pip install -e '.[speech,dev]'
pytest tests/unit/test_speech.py                 # the speech path, with fake models
echovector speech index path/to/shiurim          # transcribe once, then embed
echovector speech search "the threshold at the entrance"
```

The speech code is in `echovector/speech/`. Open an issue or a draft PR describing the
experiment and the numbers you measured.
