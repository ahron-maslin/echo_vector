import os
from typing import Optional
import librosa
import numpy as np
import numpy.typing as npt


class AudioProcessor:
    """Processes audio files for vectorization."""

    def __init__(self, target_sample_rate: int = 16000, mono: bool = True) -> None:
        """Initialize the AudioProcessor.

        Args:
            target_sample_rate: The sample rate to convert audio to.
            mono: Whether to convert audio to mono.
        """
        self.target_sample_rate = target_sample_rate
        self.mono = mono

    def load_audio(self, file_path: str) -> npt.NDArray[np.float32]:
        """Load an audio file into a numpy array.

        Args:
            file_path: Path to the audio file (mp3/wav/flac/m4a).

        Returns:
            The loaded audio as a numpy array.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        audio: npt.NDArray[np.float32]
        audio, _ = librosa.load(
            file_path,
            sr=self.target_sample_rate,
            mono=self.mono
        )
        return audio
