import os

import numpy as np
import numpy.typing as npt
import soundfile as sf

from echovector.audio import dsp
from echovector.audio.decode import decode_with_fallback


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

        try:
            audio, sample_rate = sf.read(file_path, dtype="float32", always_2d=False)
        except sf.LibsndfileError:
            audio, sample_rate = decode_with_fallback(file_path)

        if self.mono and audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sample_rate != self.target_sample_rate:
            audio = dsp.resample(
                np.asarray(audio, dtype=np.float32), int(sample_rate), self.target_sample_rate
            )

        return np.asarray(audio, dtype=np.float32)
