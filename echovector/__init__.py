"""EchoVector: Audio vector embedding and processing library."""

import warnings

from echovector.core import EchoVector

# librosa falls back to its deprecated audioread loader for formats libsndfile
# can't decode natively (e.g. m4a/aac), emitting a noisy FutureWarning we can't
# act on from here.
warnings.filterwarnings(
    "ignore",
    message="librosa.core.audio.__audioread_load",
    category=FutureWarning,
)

__version__ = "0.1.0"

__all__ = ["EchoVector"]
