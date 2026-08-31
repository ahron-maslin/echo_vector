"""EchoVector: semantic text search over audio files without full transcription."""

import warnings
from importlib.metadata import PackageNotFoundError, version

from echovector.core import EchoVector
from echovector.embeddings.base import TextSearchUnsupportedError
from echovector.indexing.manifest import IndexMismatchError
from echovector.search.filters import SearchFilter
from echovector.search.results import SearchResult, TimestampRange

# librosa falls back to its deprecated audioread loader for formats libsndfile
# can't decode natively (e.g. m4a/aac), emitting a noisy FutureWarning we can't
# act on from here.
warnings.filterwarnings(
    "ignore",
    message="librosa.core.audio.__audioread_load",
    category=FutureWarning,
)

try:
    # Single source of truth: the installed distribution metadata, which hatch-vcs
    # derives from the git tag. Hardcoding it here drifts from the released version.
    __version__ = version("echo_vector")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "EchoVector",
    "IndexMismatchError",
    "SearchFilter",
    "SearchResult",
    "TextSearchUnsupportedError",
    "TimestampRange",
    "__version__",
]
