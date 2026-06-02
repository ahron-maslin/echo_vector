"""
Pytest configuration and shared fixtures for tests.
"""
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.audio_generators import generate_sine_wave, save_audio


@pytest.fixture
def temp_audio_dir() -> Generator[Path, None, None]:
    """
    Fixture providing a temporary directory for audio files.

    Yields:
        Path object pointing to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

@pytest.fixture
def synthetic_sine_audio(temp_audio_dir: Path) -> Path:
    """
    Fixture generating a synthetic sine wave audio file and returning its path.

    Args:
        temp_audio_dir: Temporary directory path.

    Returns:
        Path to the generated temporary .wav file.
    """
    audio_path = temp_audio_dir / "sine_440hz.wav"
    sample_rate = 16000
    audio_data = generate_sine_wave(frequency=440.0, duration=0.5, sample_rate=sample_rate)
    save_audio(audio_path, audio_data, sample_rate)
    return audio_path

@pytest.fixture
def stereo_sine_audio(temp_audio_dir: Path) -> Path:
    """
    Fixture generating a synthetic stereo sine wave audio file.

    Args:
        temp_audio_dir: Temporary directory path.

    Returns:
        Path to the generated temporary stereo .wav file.
    """
    audio_path = temp_audio_dir / "stereo_sine.wav"
    sample_rate = 16000
    left_channel = generate_sine_wave(frequency=440.0, duration=0.5, sample_rate=sample_rate)
    right_channel = generate_sine_wave(frequency=880.0, duration=0.5, sample_rate=sample_rate)
    stereo_data = np.vstack((left_channel, right_channel)).T
    save_audio(audio_path, stereo_data, sample_rate)
    return audio_path
