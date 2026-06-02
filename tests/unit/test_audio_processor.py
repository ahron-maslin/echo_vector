import os
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
    fd, path = tempfile.mkstemp(suffix='.wav')
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


@patch('echovector.audio.metadata.sf.info')
@patch('echovector.audio.metadata.librosa')
def test_extract_metadata_fallback(mock_librosa: patch, mock_sf_info: patch, temp_audio_file: str) -> None:
    mock_sf_info.side_effect = Exception("soundfile failed")

    mock_librosa.get_duration.return_value = 1.5
    mock_librosa.get_samplerate.return_value = 16000

    mock_y = np.zeros((2, 16000))
    mock_librosa.load.return_value = (mock_y, 16000)

    metadata = extract_metadata(temp_audio_file)

    assert metadata.channels == 2
    assert metadata.sample_rate == 16000


def test_audio_streamer(temp_audio_file: str) -> None:
    streamer = AudioStreamer(block_size=4000)
    blocks = list(streamer.stream(temp_audio_file))

    assert len(blocks) == 6 # 16000 * 1.5 = 24000, 24000 / 4000 = 6
    assert all(isinstance(b, np.ndarray) for b in blocks)
    assert all(len(b) == 4000 for b in blocks)


def test_audio_streamer_stereo_to_mono() -> None:
    fd, path = tempfile.mkstemp(suffix='.wav')
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
