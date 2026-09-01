import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from echovector.audio.metadata import AudioMetadata, extract_metadata
from echovector.audio.processor import AudioProcessor
from echovector.audio.streaming import AudioStreamer


@pytest.fixture
def temp_audio_file() -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    # Generate some silence
    sr = 16000
    duration = 1.5
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(path, samples, sr)

    yield path

    os.remove(path)


def test_audio_processor_load(temp_audio_file: str) -> None:
    processor = AudioProcessor(target_sample_rate=8000, mono=True)
    audio = processor.load_audio(temp_audio_file)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == int(8000 * 1.5)


def test_audio_processor_not_found() -> None:
    processor = AudioProcessor()
    with pytest.raises(FileNotFoundError):
        processor.load_audio("nonexistent_file.wav")


def test_extract_metadata(temp_audio_file: str) -> None:
    metadata = extract_metadata(temp_audio_file)

    assert isinstance(metadata, AudioMetadata)
    assert metadata.sample_rate == 16000
    assert pytest.approx(metadata.duration, 0.1) == 1.5
    assert metadata.channels == 1
    assert metadata.file_path == temp_audio_file


def test_extract_metadata_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        extract_metadata("nonexistent_file.wav")


@patch("echovector.audio.processor.decode_with_fallback")
@patch("echovector.audio.processor.sf")
def test_audio_processor_uses_the_fallback_decoder(
    mock_sf: patch, mock_decode: patch, temp_audio_file: str
) -> None:
    """Stereo audio from the fallback decoder should downmix across channels.

    The fallback returns (samples, channels), matching soundfile, so the
    downmix must average across channels rather than time.
    """
    mock_sf.LibsndfileError = sf.LibsndfileError
    mock_sf.read.side_effect = sf.LibsndfileError("unsupported format")

    stereo = np.stack([np.full(1000, 1.0), np.full(1000, 3.0)], axis=1).astype(np.float32)
    mock_decode.return_value = (stereo, 16000)

    processor = AudioProcessor(target_sample_rate=16000, mono=True)
    audio = processor.load_audio(temp_audio_file)

    assert audio.ndim == 1
    assert audio.shape[0] == 1000
    assert np.allclose(audio, 2.0)


@patch("echovector.audio.metadata.sf.info")
@patch("echovector.audio.metadata.decode_with_fallback")
def test_extract_metadata_fallback(
    mock_decode: patch, mock_sf_info: patch, temp_audio_file: str
) -> None:
    mock_sf_info.side_effect = Exception("soundfile failed")
    mock_decode.return_value = (np.zeros((16000, 2), dtype=np.float32), 16000)

    metadata = extract_metadata(temp_audio_file)

    assert metadata.channels == 2
    assert metadata.sample_rate == 16000
    assert metadata.duration == pytest.approx(1.0)


def test_fallback_decoder_explains_the_missing_extra(monkeypatch) -> None:
    """Without librosa, an exotic format must say how to make it work."""
    import builtins

    from echovector.audio.decode import FALLBACK_EXTRA, decode_with_fallback

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == "librosa":
            raise ModuleNotFoundError("No module named 'librosa'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "librosa", raising=False)

    with pytest.raises(ImportError, match=f"echo_vector\\[{FALLBACK_EXTRA}\\]"):
        decode_with_fallback("weird.m4a")


def test_audio_streamer(temp_audio_file: str) -> None:
    streamer = AudioStreamer(block_size=4000)
    blocks = list(streamer.stream(temp_audio_file))

    assert len(blocks) == 6  # 16000 * 1.5 = 24000, 24000 / 4000 = 6
    assert all(isinstance(b, np.ndarray) for b in blocks)
    assert all(len(b) == 4000 for b in blocks)


def test_audio_streamer_stereo_to_mono() -> None:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    sr = 16000
    duration = 1.0
    samples = np.zeros((int(sr * duration), 2), dtype=np.float32)
    sf.write(path, samples, sr)

    try:
        streamer = AudioStreamer(block_size=8000)
        blocks = list(streamer.stream(path))

        assert len(blocks) == 2
        assert blocks[0].ndim == 1
    finally:
        os.remove(path)
