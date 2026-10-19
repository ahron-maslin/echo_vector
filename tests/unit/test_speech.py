from pathlib import Path

import numpy as np
import pytest

from echovector.speech import (
    SpeechIndex,
    Word,
    build_vocab,
    chunk_words,
    expand_query,
    load_transcript,
    save_transcript,
)


def words(text: str, step: float = 1.0) -> list[Word]:
    return [Word(w, i * step, (i + 1) * step) for i, w in enumerate(text.split())]


def fake_ipa(surfaces: list[str]) -> dict[str, str]:
    table = {"escoopa": "eskupa", "scuba": "skuba", "is scuba": "iskuba", "table": "teibel"}
    return {s: table.get(s, s) for s in surfaces}


class TestChunking:
    def test_windows_overlap_and_carry_times(self) -> None:
        chunks = chunk_words(words(" ".join(f"w{i}" for i in range(10))), size=4, stride=2)
        assert chunks[0].text == "w0 w1 w2 w3"
        assert chunks[1].text.startswith("w2")
        assert (chunks[1].start, chunks[1].end) == (2.0, 6.0)

    def test_short_transcript_yields_one_chunk(self) -> None:
        assert len(chunk_words(words("a b c"), size=45, stride=25)) == 1

    def test_empty_transcript_yields_nothing(self) -> None:
        assert chunk_words([]) == []

    def test_blank_words_are_dropped(self) -> None:
        chunks = chunk_words([Word(" ", 0, 1), Word("hi", 1, 2)])
        assert chunks[0].text == "hi"


class TestTranscriptCache:
    def test_roundtrip(self, tmp_path: Path) -> None:
        original = words("hello world")
        path = tmp_path / "t.json"
        save_transcript(path, original)
        assert load_transcript(path) == original


class TestPhoneticExpansion:
    def test_vocab_has_unigrams_and_bigrams(self) -> None:
        vocab = build_vocab(words("is scuba fun"))
        assert {"is", "scuba", "fun", "is scuba", "scuba fun"} <= set(vocab)

    def test_frequent_near_homophones_expand_the_query(self) -> None:
        text = "escoopa " + "scuba " * 8 + "table"
        vocab = build_vocab(words(text))
        found = expand_query("escoopa", vocab, fake_ipa, query_ipa="eskupa", top_k=3)
        assert found[0] == "escoopa"
        assert "scuba" in found

    def test_unrelated_words_are_not_added(self) -> None:
        vocab = build_vocab(words("table table table"))
        found = expand_query("escoopa", vocab, fake_ipa, query_ipa="eskupa", top_k=3, floor=0.6)
        assert "table" not in found

    def test_query_always_leads(self) -> None:
        assert expand_query("zzz", {}, fake_ipa, query_ipa="zzz")[0] == "zzz"


class KeywordEmbedder:
    """Two axes: 'threshold' topic and 'tax' topic. Stands in for a real model."""

    def __call__(self, texts: list[str], *, query: bool) -> np.ndarray:
        out = np.zeros((len(texts), 2), dtype=np.float32)
        for i, t in enumerate(texts):
            low = t.lower()
            out[i, 0] = sum(low.count(k) for k in ("threshold", "escoopa", "scuba", "entrance"))
            out[i, 1] = sum(low.count(k) for k in ("tax", "money"))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


@pytest.fixture
def speech_index(tmp_path: Path) -> SpeechIndex:
    index = SpeechIndex(
        tmp_path / "idx", embedder=KeywordEmbedder(), ipa=fake_ipa, chunk_size=4, stride=4
    )
    index.add("a.mp3", words("the escoopa is the threshold here scuba scuba scuba"))
    index.add("b.mp3", words("pay the tax and the money now please do it"))
    return index


class TestSpeechIndex:
    def test_concept_query_lands_on_right_file(self, speech_index: SpeechIndex) -> None:
        hits = speech_index.search("the entrance", k=2)
        assert hits[0].file == "a.mp3"

    def test_hit_carries_timestamps_and_text(self, speech_index: SpeechIndex) -> None:
        hit = speech_index.search("tax money", k=1)[0]
        assert hit.file == "b.mp3"
        assert hit.end > hit.start >= 0
        assert "tax" in hit.text or "money" in hit.text

    def test_expansion_finds_misspelled_term(self, speech_index: SpeechIndex) -> None:
        hit = speech_index.search("escoopa", k=1, query_ipa="eskupa")[0]
        assert hit.file == "a.mp3"

    def test_persists_across_reopen(self, speech_index: SpeechIndex, tmp_path: Path) -> None:
        speech_index.save()
        reopened = SpeechIndex(tmp_path / "idx", embedder=KeywordEmbedder(), ipa=fake_ipa)
        assert reopened.search("tax", k=1)[0].file == "b.mp3"

    def test_readding_a_file_replaces_it(self, speech_index: SpeechIndex) -> None:
        speech_index.add("b.mp3", words("threshold threshold threshold threshold"))
        assert all(
            h.file != "b.mp3" or "threshold" in h.text for h in speech_index.search("tax", k=5)
        )

    def test_empty_index_returns_nothing(self, tmp_path: Path) -> None:
        index = SpeechIndex(tmp_path / "e", embedder=KeywordEmbedder(), ipa=fake_ipa)
        assert index.search("anything") == []
