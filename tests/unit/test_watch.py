"""Tests for the folder watcher.

`scan_once` does the work and `run` only loops, so almost everything is tested
by driving scans directly — no sleeping, no timing races.
"""

import os
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from echovector import EchoVector
from echovector.watch import FolderWatcher, WatchEvent


def _write_tone(path: Path, frequency: float = 440.0, duration: float = 2.0) -> None:
    sample_rate = 16_000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sf.write(path, (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sample_rate)


@pytest.fixture
def library(tmp_path: Path):
    audio_dir = tmp_path / "library"
    audio_dir.mkdir()
    engine = EchoVector(
        store_dir=tmp_path / "store",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=16_000,
    )
    yield engine, audio_dir
    engine.close()


def _kinds(events: list[WatchEvent]) -> list[str]:
    return [event.kind for event in events]


def test_new_files_are_indexed(library) -> None:
    engine, audio_dir = library
    watcher = FolderWatcher(engine, [audio_dir], settle_scans=0)
    _write_tone(audio_dir / "bird.wav")

    events = watcher.scan_once()

    assert _kinds(events) == ["added"]
    assert events[0].chunks == 2
    assert engine.stats()["chunks"] == 2


def test_an_unchanged_library_produces_no_events(library) -> None:
    engine, audio_dir = library
    watcher = FolderWatcher(engine, [audio_dir], settle_scans=0)
    _write_tone(audio_dir / "bird.wav")
    watcher.scan_once()

    assert watcher.scan_once() == []


def test_edited_files_are_reindexed_as_changed(library) -> None:
    engine, audio_dir = library
    watcher = FolderWatcher(engine, [audio_dir], settle_scans=0)
    path = audio_dir / "bird.wav"
    _write_tone(path, duration=2.0)
    watcher.scan_once()

    _write_tone(path, frequency=880.0, duration=4.0)
    os.utime(path, (0, 0))
    events = watcher.scan_once()

    assert _kinds(events) == ["changed"]
    assert engine.stats()["chunks"] == 4
    assert engine.index_backend.store.count_files() == 1


def test_deleted_files_are_dropped(library) -> None:
    engine, audio_dir = library
    watcher = FolderWatcher(engine, [audio_dir], settle_scans=0)
    _write_tone(audio_dir / "keep.wav")
    _write_tone(audio_dir / "gone.wav", frequency=880.0)
    watcher.scan_once()

    (audio_dir / "gone.wav").unlink()
    events = watcher.scan_once()

    assert _kinds(events) == ["removed"]
    assert events[0].filepath.name == "gone.wav"
    assert engine.index_backend.store.count_files() == 1
    assert engine.index_backend.desync_report() is None


def test_a_file_still_being_written_is_left_alone(library) -> None:
    """Indexing a half-copied file captures a truncated recording.

    Worse, it then looks complete: the stored signature matches what was read,
    so nothing would ever re-index it.
    """
    engine, audio_dir = library
    watcher = FolderWatcher(engine, [audio_dir], settle_scans=1)
    path = audio_dir / "growing.wav"
    _write_tone(path, duration=2.0)

    # First sighting: signature not yet confirmed stable, so nothing is indexed.
    assert watcher.scan_once() == []
    assert engine.stats()["chunks"] == 0

    # Still growing.
    _write_tone(path, duration=4.0)
    os.utime(path, (0, 0))
    assert watcher.scan_once() == []
    assert engine.stats()["chunks"] == 0

    # Now stable across two scans, so it is indexed whole.
    events = watcher.scan_once()
    assert _kinds(events) == ["added"]
    assert engine.stats()["chunks"] == 4


def test_settle_state_is_dropped_when_a_file_vanishes(library) -> None:
    engine, audio_dir = library
    watcher = FolderWatcher(engine, [audio_dir], settle_scans=2)
    path = audio_dir / "transient.wav"
    _write_tone(path)
    watcher.scan_once()

    path.unlink()
    watcher.scan_once()

    assert watcher._settling == {}


def test_one_unreadable_file_does_not_stop_the_watch(library, monkeypatch) -> None:
    """A library picks up partial downloads and odd formats; keep going."""
    engine, audio_dir = library
    watcher = FolderWatcher(engine, [audio_dir], settle_scans=0)
    _write_tone(audio_dir / "a.wav")
    _write_tone(audio_dir / "b.wav", frequency=880.0)

    real_index = engine.index

    def explode_on_a(target, **kwargs):
        if Path(target).name == "a.wav":
            raise RuntimeError("corrupt header")
        return real_index(target, **kwargs)

    monkeypatch.setattr(engine, "index", explode_on_a)
    events = watcher.scan_once()

    assert sorted(_kinds(events)) == ["added", "failed"]
    failed = next(e for e in events if e.kind == "failed")
    assert "corrupt header" in (failed.error or "")


def test_run_stops_when_the_event_is_set(library) -> None:
    engine, audio_dir = library
    _write_tone(audio_dir / "bird.wav")
    watcher = FolderWatcher(engine, [audio_dir], settle_scans=0)

    stop = threading.Event()
    seen: list[WatchEvent] = []
    scans = watcher.run(
        interval=0.01,
        on_event=lambda event: (seen.append(event), stop.set())[0],
        stop=stop,
    )

    assert scans == 1
    assert _kinds(seen) == ["added"]


def test_run_honours_max_scans(library) -> None:
    engine, audio_dir = library
    watcher = FolderWatcher(engine, [audio_dir], settle_scans=0)
    counted: list[int] = []

    scans = watcher.run(interval=0.01, on_scan=counted.append, max_scans=3)

    assert scans == 3
    assert len(counted) == 3


def test_run_rejects_a_non_positive_interval(library) -> None:
    engine, audio_dir = library
    with pytest.raises(ValueError, match="interval must be positive"):
        FolderWatcher(engine, [audio_dir]).run(interval=0)


def test_non_recursive_watching(library) -> None:
    engine, audio_dir = library
    (audio_dir / "nested").mkdir()
    _write_tone(audio_dir / "top.wav")
    _write_tone(audio_dir / "nested" / "deep.wav", frequency=880.0)

    watcher = FolderWatcher(engine, [audio_dir], recursive=False, settle_scans=0)
    events = watcher.scan_once()

    assert [e.filepath.name for e in events] == ["top.wav"]
