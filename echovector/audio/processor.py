import os

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf


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
            audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
            if audio.ndim > 1:
                # librosa returns (channels, samples); soundfile returns (samples, channels).
                audio = audio.T

        if self.mono and audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sample_rate != self.target_sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=int(sample_rate),
                target_sr=self.target_sample_rate,
                axis=0,
            )

        return np.asarray(audio, dtype=np.float32)
