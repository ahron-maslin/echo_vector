"""
Synthetic audio generators for testing.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf


def generate_sine_wave(
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> npt.NDArray[np.float32]:
    """
    Generate a synthetic sine wave audio signal.

    Args:
        frequency: Frequency of the sine wave in Hz.
        duration: Duration of the signal in seconds.
        sample_rate: Sampling rate in Hz.
        amplitude: Amplitude of the signal (0.0 to 1.0).

    Returns:
        1D numpy array containing the generated audio signal.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal.astype(np.float32)


def save_audio(
    file_path: str | Path, audio_data: npt.NDArray[np.float32], sample_rate: int = 16000
) -> None:
    """
    Save audio data to a file using soundfile.

    Args:
        file_path: Path to save the audio file (e.g., .wav).
        audio_data: Numpy array containing audio data.
        sample_rate: Sampling rate of the audio data.
    """
    sf.write(file_path, audio_data, sample_rate)
