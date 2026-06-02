import os
from dataclasses import dataclass

import librosa
import soundfile as sf


@dataclass
class AudioMetadata:
    """Metadata for an audio file."""
    duration: float
    sample_rate: int
    channels: int
    format: str
    file_size: int
    file_path: str


def extract_metadata(file_path: str) -> AudioMetadata:
    """Extract metadata from an audio file.

    Args:
        file_path: Path to the audio file.

    Returns:
        AudioMetadata object containing extracted metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    file_size = os.path.getsize(file_path)

    try:
        info = sf.info(file_path)
        duration = float(info.duration)
        sample_rate = int(info.samplerate)
        channels = int(info.channels)
        fmt = str(info.format)
    except Exception:
        # Fallback to librosa
        duration = float(librosa.get_duration(path=file_path))
        sample_rate = int(librosa.get_samplerate(file_path))
        y, _ = librosa.load(file_path, sr=None, mono=False)
        channels = int(y.shape[0]) if y.ndim > 1 else 1
        fmt = str(os.path.splitext(file_path)[1].lstrip('.'))

    return AudioMetadata(
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
        format=fmt,
        file_size=file_size,
        file_path=file_path
    )
