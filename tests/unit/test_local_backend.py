"""Unit tests for the deterministic LocalFeatureBackend."""

import numpy as np
import soundfile as sf

from echovector.embeddings.local import LocalFeatureBackend


def _write_tone(path, sample_rate=16_000, frequency=440.0, duration=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sf.write(path, (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sample_rate)


def test_embedding_dim():
    assert LocalFeatureBackend().embedding_dim == 8


def test_embed_audio_resamples_when_rate_differs(tmp_path):
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path, sample_rate=8000)

    backend = LocalFeatureBackend(sample_rate=16_000)
    embedding = backend.embed_audio([str(audio_path)])

    assert embedding.shape == (1, 8)
    assert np.isfinite(embedding).all()


def test_embed_audio_silence_returns_zero_vector(tmp_path):
    audio_path = tmp_path / "silence.wav"
    sf.write(audio_path, np.zeros(0, dtype=np.float32), 16_000)

    backend = LocalFeatureBackend()
    embedding = backend.embed_audio([str(audio_path)])

    assert embedding.shape == (1, 8)
    np.testing.assert_array_equal(embedding[0], np.zeros(8, dtype=np.float32))


def test_embed_text_keyword_branches():
    backend = LocalFeatureBackend()
    keywords = [
        "a long extended recording",
        "a loud alarm",
        "noisy busy speech",
        "a high bright alert",
        "wide broadband noise",
        "flat static noise",
        "totally silent and quiet",
        "a low deep bass tone",
        "no keywords here",
    ]
    embeddings = backend.embed_text(keywords)

    assert embeddings.shape == (len(keywords), 8)
    assert np.isfinite(embeddings).all()
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
