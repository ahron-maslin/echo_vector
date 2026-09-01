"""Keep an index current as a folder changes.

Indexing is normally something you remember to run. A watcher makes it something
you forget about: point it at a library and it re-indexes what changed, adds what
appeared, and drops what was deleted.

Deliberately polls rather than using filesystem events. Watching an audio library
means watching network shares, external drives, and sync clients, where native
events are unreliable or absent; polling a signature the store already records is
predictable everywhere and adds no dependency to an install that was just cut
down to size.
"""

import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from echovector.core import EchoVector

__all__ = ["DEFAULT_INTERVAL_SECONDS", "FolderWatcher", "WatchEvent"]

#: How often to rescan, in seconds. Long enough not to thrash a large library,
#: short enough that a new recording is searchable while you still care.
DEFAULT_INTERVAL_SECONDS = 10.0

EventKind = Literal["added", "changed", "removed", "failed"]


@dataclass(frozen=True)
class WatchEvent:
    """One thing the watcher did, or failed to do.

    Attributes:
        kind: What happened to the file.
        filepath: The audio file concerned.
        chunks: Chunks added, for "added" and "changed".
        error: Why indexing failed, for "failed".
    """

    kind: EventKind
    filepath: Path
    chunks: int = 0
    error: str | None = None


class FolderWatcher:
    """Rescans targets on an interval and keeps an index in step with them."""

    def __init__(
        self,
        engine: EchoVector,
        targets: Sequence[str | Path],
        recursive: bool | None = None,
        settle_scans: int = 1,
    ) -> None:
        """Initialize the watcher.

        Args:
            engine: The engine whose index is kept current.
            targets: Files or directories to watch.
            recursive: Override the engine's recursive setting.
            settle_scans: Consecutive scans a file's signature must hold steady
                before it is indexed. Guards against reading a file that is
                still being written; 0 indexes immediately.
        """
        self._engine = engine
        self._targets = list(targets)
        self._recursive = recursive
        self._settle_scans = max(0, settle_scans)
        # filepath -> (signature, consecutive scans it has looked identical)
        self._settling: dict[Path, tuple[tuple[float, int], int]] = {}

    def scan_once(self) -> list[WatchEvent]:
        """Bring the index in line with the current state of the targets.

        Returns:
            What changed, in the order it was handled.
        """
        events: list[WatchEvent] = [
            WatchEvent(kind="removed", filepath=Path(path)) for path in self._engine.prune()
        ]

        present = self._engine.resolve_audio_files(self._targets, recursive=self._recursive)
        self._forget_vanished(present)

        for file_path in present:
            if not self._engine.is_stale(file_path):
                self._settling.pop(file_path, None)
                continue
            if not self._has_settled(file_path):
                continue

            was_indexed = self._engine.is_indexed(file_path)
            self._settling.pop(file_path, None)
            events.append(self._index_one(file_path, was_indexed))

        return events

    def run(
        self,
        interval: float = DEFAULT_INTERVAL_SECONDS,
        on_event: Callable[[WatchEvent], None] | None = None,
        on_scan: Callable[[int], None] | None = None,
        stop: threading.Event | None = None,
        max_scans: int | None = None,
    ) -> int:
        """Scan repeatedly until stopped.

        Args:
            interval: Seconds between scans.
            on_event: Called for each change, as it happens.
            on_scan: Called after each scan with the number of files watched.
            stop: Set this to end the loop; also used to sleep, so shutdown is
                immediate rather than waiting out the interval.
            max_scans: Stop after this many scans. Mainly for tests.

        Returns:
            The number of scans performed.

        Raises:
            ValueError: If interval is not positive.
        """
        if interval <= 0:
            raise ValueError("interval must be positive.")

        stop = stop or threading.Event()
        scans = 0
        while not stop.is_set():
            for event in self.scan_once():
                if on_event is not None:
                    on_event(event)
            scans += 1
            if on_scan is not None:
                on_scan(len(self._engine.resolve_audio_files(self._targets, self._recursive)))

            if max_scans is not None and scans >= max_scans:
                break
            # Waiting on the event rather than sleeping means Ctrl-C is felt at
            # once instead of after the remainder of the interval.
            stop.wait(interval)
        return scans

    def _index_one(self, file_path: Path, was_indexed: bool) -> WatchEvent:
        """Index a single file, turning any failure into an event.

        One unreadable file must not end the watch; a library picks up partial
        downloads and odd formats, and the watcher has to survive them.
        """
        kind: EventKind = "changed" if was_indexed else "added"
        try:
            chunks = self._engine.index(file_path, force=True)
        except Exception as exc:  # reported as an event, never fatal to the watch
            return WatchEvent(kind="failed", filepath=file_path, error=str(exc))
        return WatchEvent(kind=kind, filepath=file_path, chunks=chunks)

    def _has_settled(self, file_path: Path) -> bool:
        """Return True once a file has stopped changing.

        A file being copied in grows between scans. Indexing it then captures a
        truncated recording and, worse, looks complete afterwards because the
        stored signature matches whatever was read.
        """
        if self._settle_scans == 0:
            return True
        try:
            stat = file_path.stat()
        except OSError:
            return False

        signature = (stat.st_mtime, stat.st_size)
        previous, stable_scans = self._settling.get(file_path, (None, 0))
        stable_scans = stable_scans + 1 if previous == signature else 0
        self._settling[file_path] = (signature, stable_scans)
        return stable_scans >= self._settle_scans

    def _forget_vanished(self, present: Iterable[Path]) -> None:
        """Drop settle state for files that are no longer there."""
        present_set = set(present)
        for path in [p for p in self._settling if p not in present_set]:
            del self._settling[path]


def watch(
    engine: EchoVector,
    targets: Sequence[str | Path],
    interval: float = DEFAULT_INTERVAL_SECONDS,
    on_event: Callable[[WatchEvent], None] | None = None,
) -> None:
    """Watch targets until interrupted.

    Args:
        engine: The engine whose index is kept current.
        targets: Files or directories to watch.
        interval: Seconds between scans.
        on_event: Called for each change.
    """
    watcher = FolderWatcher(engine, targets)
    stop = threading.Event()
    try:
        watcher.run(interval=interval, on_event=on_event, stop=stop)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        stop.set()
