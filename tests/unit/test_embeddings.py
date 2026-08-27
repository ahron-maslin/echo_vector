"""
Unit tests for the EchoVector embeddings module.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from echovector.embeddings.ast_model import ASTBackend
from echovector.embeddings.base import (
    WaveformEmbeddingBackend,
    backend_fingerprint,
    max_audio_seconds,
)
from echovector.embeddings.clap import ClapBackend
from echovector.embeddings.factory import (
    EmbeddingFactory,
    backend_class_for,
    get_embedding_model,
)
from echovector.embeddings.hubert import HubertBackend
from echovector.embeddings.local import LocalFeatureBackend
from echovector.embeddings.wav2vec2 import Wav2Vec2Backend


def _mock_clap(mock_processor_class, mock_model_class, projection_dim=512):
    mock_processor = MagicMock()
    mock_processor_class.from_pretrained.return_value = mock_processor
    mock_processor.feature_extractor.sampling_rate = 48000
    mock_processor.feature_extractor.max_length_s = 10

    mock_model = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model
    mock_model.config.projection_dim = projection_dim
    return mock_processor, mock_model


def _mock_features(mock_model, attr, array):
    features = MagicMock()
    features.norm.return_value = 1.0
    features.__truediv__.return_value = features
    features.cpu.return_value.numpy.return_value = array
    getattr(mock_model, attr).return_value = features
    return features


class TestClapBackend(unittest.TestCase):
    @patch("echovector.embeddings.clap.ClapModel")
    @patch("echovector.embeddings.clap.ClapProcessor")
    @patch("echovector.embeddings.clap.librosa")
    def test_embed_audio(self, mock_librosa, mock_processor_class, mock_model_class):
        """Test audio embedding via CLAP without downloading models."""
        _, mock_model = _mock_clap(mock_processor_class, mock_model_class)
        _mock_features(mock_model, "get_audio_features", np.zeros((1, 512), dtype=np.float32))
        mock_librosa.load.return_value = (np.zeros(48000), 48000)

        backend = ClapBackend(device="cpu")
        embeddings = backend.embed_audio(["dummy.wav"])

        assert embeddings.shape == (1, 512)
        assert backend.embedding_dim == 512
        mock_librosa.load.assert_called_once()
        mock_model.get_audio_features.assert_called_once()

    @patch("echovector.embeddings.clap.ClapModel")
    @patch("echovector.embeddings.clap.ClapProcessor")
    def test_embed_text(self, mock_processor_class, mock_model_class):
        """Test text embedding via CLAP without downloading models."""
        _, mock_model = _mock_clap(mock_processor_class, mock_model_class)
        _mock_features(mock_model, "get_text_features", np.ones((2, 512), dtype=np.float32))

        backend = ClapBackend(device="cpu")
        embeddings = backend.embed_text(["hello", "world"])

        assert embeddings.shape == (2, 512)
        mock_model.get_text_features.assert_called_once()

    @patch("echovector.embeddings.clap.ClapModel")
    @patch("echovector.embeddings.clap.ClapProcessor")
    @patch("echovector.embeddings.clap.librosa")
    def test_embed_audio_arrays_skips_file_io(
        self, mock_librosa, mock_processor_class, mock_model_class
    ):
        """Waveforms already in memory must not be routed back through librosa."""
        _, mock_model = _mock_clap(mock_processor_class, mock_model_class)
        _mock_features(mock_model, "get_audio_features", np.zeros((2, 512), dtype=np.float32))

        backend = ClapBackend(device="cpu")
        waveforms = [np.zeros(48000, dtype=np.float32), np.zeros(24000, dtype=np.float32)]
        embeddings = backend.embed_audio_arrays(waveforms, sample_rate=48000)

        assert embeddings.shape == (2, 512)
        mock_librosa.load.assert_not_called()
        mock_librosa.resample.assert_not_called()

    @patch("echovector.embeddings.clap.ClapModel")
    @patch("echovector.embeddings.clap.ClapProcessor")
    @patch("echovector.embeddings.clap.librosa")
    def test_embed_audio_arrays_resamples_when_needed(
        self, mock_librosa, mock_processor_class, mock_model_class
    ):
        """A batch at the wrong sample rate is resampled to the model's rate."""
        _, mock_model = _mock_clap(mock_processor_class, mock_model_class)
        _mock_features(mock_model, "get_audio_features", np.zeros((1, 512), dtype=np.float32))
        mock_librosa.resample.return_value = np.zeros(48000, dtype=np.float32)

        backend = ClapBackend(device="cpu")
        backend.embed_audio_arrays([np.zeros(16000, dtype=np.float32)], sample_rate=16000)

        mock_librosa.resample.assert_called_once()

    @patch("echovector.embeddings.clap.ClapModel")
    @patch("echovector.embeddings.clap.ClapProcessor")
    def test_reports_fingerprint_and_audio_window(self, mock_processor_class, mock_model_class):
        """The manifest needs the exact weights and the model's audio window."""
        _mock_clap(mock_processor_class, mock_model_class)

        backend = ClapBackend(model_name="laion/clap-htsat-unfused", device="cpu")

        assert backend.fingerprint == "clap:laion/clap-htsat-unfused"
        assert backend.max_audio_seconds == 10.0
        assert backend.target_sample_rate == 48000

    @patch("echovector.embeddings.clap.ClapModel")
    @patch("echovector.embeddings.clap.ClapProcessor")
    def test_falls_back_when_feature_extractor_lacks_settings(
        self, mock_processor_class, mock_model_class
    ):
        """A processor without the expected attributes must not break embedding."""
        mock_processor = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        del mock_processor.feature_extractor
        mock_model_class.from_pretrained.return_value.config.projection_dim = 512

        backend = ClapBackend(device="cpu")

        assert backend.target_sample_rate == 48000
        assert backend.max_audio_seconds == 10.0


class TestLocalBackendArrays:
    def test_array_and_file_paths_agree(self, tmp_path):
        """Embedding a waveform must match embedding the same audio from disk."""
        import soundfile as sf

        sample_rate = 16000
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        wave = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        path = tmp_path / "tone.wav"
        # Written as FLOAT: the default PCM_16 subtype would quantize the
        # reference and the two paths would legitimately disagree.
        sf.write(path, wave, sample_rate, subtype="FLOAT")

        backend = LocalFeatureBackend(sample_rate=sample_rate)
        from_disk = backend.embed_audio([str(path)])
        from_memory = backend.embed_audio_arrays([wave], sample_rate=sample_rate)

        np.testing.assert_allclose(from_disk, from_memory, rtol=1e-5, atol=1e-6)

    def test_empty_batches_return_correct_shape(self):
        backend = LocalFeatureBackend()
        assert backend.embed_audio([]).shape == (0, 8)
        assert backend.embed_audio_arrays([], sample_rate=16000).shape == (0, 8)
        assert backend.embed_text([]).shape == (0, 8)

    def test_declares_waveform_protocol(self):
        assert isinstance(LocalFeatureBackend(), WaveformEmbeddingBackend)

    def test_fingerprint_tracks_sample_rate(self):
        assert backend_fingerprint(LocalFeatureBackend(sample_rate=16000)) != backend_fingerprint(
            LocalFeatureBackend(sample_rate=22050)
        )

    def test_no_audio_window_limit(self):
        assert max_audio_seconds(LocalFeatureBackend()) is None


class TestBackendIntrospection:
    def test_fingerprint_defaults_to_class_name(self):
        class Anonymous:
            embedding_dim = 4

        assert backend_fingerprint(Anonymous()) == "Anonymous"

    def test_stub_backends_are_not_waveform_backends(self):
        assert not isinstance(HubertBackend(), WaveformEmbeddingBackend)


class TestEmbeddingFactory(unittest.TestCase):
    def test_create_stub_backend(self):
        """Test instantiating a stub backend via factory."""
        backend = get_embedding_model("whisper")
        assert backend.embedding_dim == 512
        with pytest.raises(NotImplementedError):
            backend.embed_text(["hello"])

    def test_invalid_backend(self):
        """Test factory with invalid backend name."""
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            get_embedding_model("nonexistent")

    def test_case_insensitivity(self):
        """Test that factory accepts names ignoring case."""
        backend = get_embedding_model("WHISPER")
        assert backend.__class__.__name__ == "WhisperBackend"

    def test_resolve_returns_class_without_instantiating(self):
        """The CLI inspects constructors before paying for a model load."""
        assert backend_class_for("local") is LocalFeatureBackend

    def test_available_backends_includes_registered(self):
        class Custom:
            embedding_dim = 2

        EmbeddingFactory.register_backend("custom-test", Custom)
        try:
            assert "custom-test" in EmbeddingFactory.available_backends()
            assert "clap" in EmbeddingFactory.available_backends()
        finally:
            EmbeddingFactory._registry.pop("custom-test", None)


@pytest.mark.parametrize(
    ("backend_cls", "expected_dim"),
    [(HubertBackend, 768), (Wav2Vec2Backend, 768), (ASTBackend, 768)],
)
def test_stub_backends_raise_not_implemented(backend_cls, expected_dim):
    """Stub backends should report their dimension but refuse to embed."""
    backend = backend_cls()
    assert backend.embedding_dim == expected_dim
    with pytest.raises(NotImplementedError):
        backend.embed_audio(["dummy.wav"])
    with pytest.raises(NotImplementedError):
        backend.embed_text(["hello"])
