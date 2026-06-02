import numpy as np
import numpy.typing as npt


class SilenceAwareChunker:
    """Chunks audio based on silence."""

    def __init__(
        self,
        top_db: float = 60.0,
        min_chunk_length: float = 1.0,
        max_chunk_length: float = 10.0,
        sample_rate: int = 16000
    ) -> None:
        """Initialize the chunker.

        Args:
            top_db: The threshold (in decibels) below reference to consider as silence.
            min_chunk_length: Minimum length of a chunk in seconds.
            max_chunk_length: Maximum length of a chunk in seconds.
            sample_rate: The sample rate of the audio.
        """
        self.top_db = top_db
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.sample_rate = sample_rate

    def chunk(self, audio: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
        """Split audio into chunks based on silence.

        Args:
            audio: The audio signal to chunk.

        Returns:
            A list of audio chunks.
        """
        if len(audio) == 0:
            return []

        max_amplitude = float(np.max(np.abs(audio)))
        if max_amplitude == 0.0:
            return []

        threshold = max_amplitude * (10.0 ** (-self.top_db / 20.0))
        non_silent = np.flatnonzero(np.abs(audio) > threshold)
        if len(non_silent) == 0:
            return []

        breaks = np.where(np.diff(non_silent) > 1)[0] + 1
        intervals = [
            (int(group[0]), int(group[-1]) + 1)
            for group in np.split(non_silent, breaks)
            if len(group) > 0
        ]

        chunks: list[npt.NDArray[np.float32]] = []
        min_samples = int(self.min_chunk_length * self.sample_rate)
        max_samples = int(self.max_chunk_length * self.sample_rate)

        for start, end in intervals:
            interval_audio: npt.NDArray[np.float32] = audio[start:end]

            while len(interval_audio) > max_samples:
                chunks.append(interval_audio[:max_samples])
                interval_audio = interval_audio[max_samples:]

            keep_short_first = len(interval_audio) > 0 and len(chunks) == 0
            if len(interval_audio) >= min_samples or keep_short_first:
                chunks.append(interval_audio)

        return chunks
