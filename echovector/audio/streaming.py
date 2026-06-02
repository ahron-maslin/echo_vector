from collections.abc import Generator

import numpy as np
import numpy.typing as npt
import soundfile as sf


class AudioStreamer:
    """Streams audio data in blocks."""

    def __init__(self, block_size: int = 4096) -> None:
        """Initialize the streamer.

        Args:
            block_size: Number of frames per block.
        """
        self.block_size = block_size

    def stream(self, file_path: str) -> Generator[npt.NDArray[np.float32], None, None]:
        """Stream audio from a file.

        Args:
            file_path: Path to the audio file.

        Yields:
            Blocks of audio data as numpy arrays.
        """
        for block in sf.blocks(
            file_path, blocksize=self.block_size, dtype="float32", always_2d=False
        ):
            if block.ndim > 1:
                block = np.mean(block, axis=1, dtype=np.float32)
            yield block
