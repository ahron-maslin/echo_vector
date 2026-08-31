import numpy as np
import numpy.typing as npt


class SilenceAwareChunker:
    """Chunks audio based on silence."""

    def __init__(
        self,
        top_db: float = 60.0,
        min_chunk_length: float = 1.0,
        max_chunk_length: float = 10.0,
        sample_rate: int = 16000,
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
        return [audio[start:end] for start, end in self._chunk_spans(audio, self.sample_rate)]

    def chunk_with_timestamps(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int | None = None,
    ) -> list[tuple[float, float, npt.NDArray[np.float32]]]:
        """Split audio into chunks tagged with their position in the source audio.

        Silence between utterances is dropped, so chunk ``n + 1`` does not generally
        start where chunk ``n`` ended. The returned offsets are absolute positions in
        ``audio``, not cumulative lengths of the emitted chunks.

        Args:
            audio: The audio signal to chunk.
            sample_rate: Sample rate of ``audio``. Defaults to the instance sample rate.

        Returns:
            A list of ``(start_seconds, end_seconds, waveform)`` tuples ordered by
            start time. No waveform is longer than ``max_chunk_length``.

        Raises:
            ValueError: If the resolved sample rate is not positive.
        """
        resolved_rate = self.sample_rate if sample_rate is None else sample_rate
        if resolved_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {resolved_rate}.")

        return [
            (start / resolved_rate, end / resolved_rate, audio[start:end])
            for start, end in self._chunk_spans(audio, resolved_rate)
        ]

    def _chunk_spans(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> list[tuple[int, int]]:
        """Compute chunk boundaries as absolute sample offsets into ``audio``.

        Args:
            audio: The audio signal to chunk.
            sample_rate: Sample rate used to convert the length bounds to samples.

        Returns:
            A list of half-open ``(start_sample, end_sample)`` spans.
        """
        spans: list[tuple[int, int]] = []
        if len(audio) == 0:
            return spans

        # A zero-length ceiling would make the splitting loop below never advance.
        max_samples = max(1, int(self.max_chunk_length * sample_rate))
        min_samples = int(self.min_chunk_length * sample_rate)

        for interval_start, interval_end in self._non_silent_intervals(audio):
            start = interval_start
            while interval_end - start > max_samples:
                spans.append((start, start + max_samples))
                start += max_samples

            # A remainder shorter than min_chunk_length is dropped rather than merged
            # into its neighbour: the neighbour is separated by discarded silence, so a
            # merged span would claim a duration the merged waveform does not contain.
            if interval_end > start and (interval_end - start >= min_samples or not spans):
                spans.append((start, interval_end))

        return spans

    def _non_silent_intervals(self, audio: npt.NDArray[np.float32]) -> list[tuple[int, int]]:
        """Locate the runs of audio that sit above the silence threshold.

        Args:
            audio: The audio signal to inspect.

        Returns:
            A list of half-open ``(start_sample, end_sample)`` spans of non-silent audio.
        """
        max_amplitude = float(np.max(np.abs(audio)))
        if max_amplitude == 0.0:
            return []

        threshold = max_amplitude * (10.0 ** (-self.top_db / 20.0))
        non_silent = np.flatnonzero(np.abs(audio) > threshold)
        if len(non_silent) == 0:
            return []

        breaks = np.where(np.diff(non_silent) > 1)[0] + 1
        return [
            (int(group[0]), int(group[-1]) + 1)
            for group in np.split(non_silent, breaks)
            if len(group) > 0
        ]
