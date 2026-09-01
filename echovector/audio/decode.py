"""Decoding for formats libsndfile cannot read on its own.

soundfile covers wav, flac, ogg, and (with a recent libsndfile) mp3. m4a/aac
still needs librosa's audioread fallback, which is why librosa remains an
optional dependency rather than a removed one: most users never touch a format
that requires it, and it costs 345 MB to carry just in case.
"""

import numpy as np
import numpy.typing as npt

__all__ = ["FALLBACK_EXTRA", "decode_with_fallback"]

#: Extra that installs the fallback decoder.
FALLBACK_EXTRA = "formats"


def decode_with_fallback(file_path: str) -> tuple[npt.NDArray[np.float32], int]:
    """Decode audio libsndfile refused, using librosa if it is installed.

    Args:
        file_path: Path to the audio file.

    Returns:
        The samples — shaped (samples,) or (samples, channels) — and their rate.

    Raises:
        ImportError: If librosa is not installed, with the command to fix it.
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            f"Cannot decode {file_path}: the format needs a fallback decoder that is not "
            f"installed. Install it with:\n\n"
            f"    pip install 'echo_vector[{FALLBACK_EXTRA}]'\n\n"
            f"WAV, FLAC, and OGG need no extra. (Underlying error: {exc})"
        ) from exc

    audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
    if audio.ndim > 1:
        # librosa returns (channels, samples); soundfile returns (samples, channels).
        audio = audio.T
    return np.asarray(audio, dtype=np.float32), int(sample_rate)
