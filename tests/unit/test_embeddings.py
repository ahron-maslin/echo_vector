"""
Unit tests for the EchoVector embeddings module.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from echovector.embeddings.cache import EmbeddingCache
from echovector.embeddings.clap import ClapBackend
from echovector.embeddings.factory import get_embedding_model


class TestClapBackend(unittest.TestCase):
    @patch("echovector.embeddings.clap.ClapModel")
    @patch("echovector.embeddings.clap.ClapProcessor")
    @patch("echovector.embeddings.clap.librosa")
    def test_embed_audio(self, mock_librosa, mock_processor_class, mock_model_class):
        """Test audio embedding via CLAP without downloading models."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_processor.feature_extractor.sampling_rate = 48000

        # Setup mock model
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.config.projection_dim = 512

        # Setup audio features return
        mock_audio_features = MagicMock()
        mock_audio_features.norm.return_value = 1.0
        mock_audio_features.__truediv__.return_value = mock_audio_features
        mock_audio_features.cpu.return_value.numpy.return_value = np.zeros((1, 512), dtype=np.float32)
        mock_model.get_audio_features.return_value = mock_audio_features

        # Setup librosa
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
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        # Setup mock model
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.config.projection_dim = 512

        # Setup text features return
        mock_text_features = MagicMock()
        mock_text_features.norm.return_value = 1.0
        mock_text_features.__truediv__.return_value = mock_text_features
        mock_text_features.cpu.return_value.numpy.return_value = np.ones((2, 512), dtype=np.float32)
        mock_model.get_text_features.return_value = mock_text_features

        backend = ClapBackend(device="cpu")
        embeddings = backend.embed_text(["hello", "world"])

        assert embeddings.shape == (2, 512)
        mock_model.get_text_features.assert_called_once()


class TestEmbeddingCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache = EmbeddingCache(cache_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_text_embedding_cache(self):
        """Test storing and retrieving text embeddings in cache."""
        text = "sample text"
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        assert self.cache.get_text_embedding(text) is None
        self.cache.put_text_embedding(text, embedding)

        cached = self.cache.get_text_embedding(text)
        assert cached is not None
        np.testing.assert_array_equal(cached, embedding)

    def test_audio_embedding_cache(self):
        """Test storing and retrieving audio embeddings in cache based on file hash."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"dummy audio content")
            filepath = f.name

        try:
            embedding = np.array([0.4, 0.5, 0.6], dtype=np.float32)

            assert self.cache.get_audio_embedding(filepath) is None
            self.cache.put_audio_embedding(filepath, embedding)

            cached = self.cache.get_audio_embedding(filepath)
            assert cached is not None
            np.testing.assert_array_equal(cached, embedding)
        finally:
            os.remove(filepath)


class TestEmbeddingFactory(unittest.TestCase):
    def test_create_stub_backend(self):
        """Test instantiating a stub backend via factory."""
        backend = get_embedding_model("whisper")
        assert backend.embedding_dim == 512
        with pytest.raises(NotImplementedError):
            backend.embed_text(["hello"])

    def test_invalid_backend(self):
        """Test factory with invalid backend name."""
        with pytest.raises(ValueError):
            get_embedding_model("nonexistent")

    def test_case_insensitivity(self):
        """Test that factory accepts names ignoring case."""
        backend = get_embedding_model("WHISPER")
        assert backend.__class__.__name__ == "WhisperBackend"


if __name__ == "__main__":
    unittest.main()
