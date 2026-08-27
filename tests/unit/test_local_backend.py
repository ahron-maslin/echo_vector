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


def test_pitch_queries_separate_high_from_low_audio(tmp_path):
    """The smoke backend must actually rank, not just return something.

    Scaling the spectral centroid by Nyquist compressed all audible content
    into the bottom of the range, leaving high/low queries separated by ~0.0005
    — noise. Octave scaling is what makes the ordering mean anything.
    """
    high, low = tmp_path / "high.wav", tmp_path / "low.wav"
    _write_tone(high, frequency=880.0)
    _write_tone(low, frequency=110.0)

    backend = LocalFeatureBackend()
    high_vec, low_vec = backend.embed_audio([str(high), str(low)])

    bright = backend.embed_text(["high bright alarm tone"])[0]
    deep = backend.embed_text(["low deep bass tone"])[0]

    assert float(high_vec @ bright) > float(low_vec @ bright) + 0.05
    assert float(low_vec @ deep) > float(high_vec @ deep) + 0.05


def test_centroid_scaling_uses_the_full_range(tmp_path):
    """A mid-range tone should land near the middle, not pinned at zero."""
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path, frequency=880.0)

    centroid = LocalFeatureBackend().embed_audio([str(audio_path)])[0][3]

    assert 0.15 < float(centroid) < 0.85


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
