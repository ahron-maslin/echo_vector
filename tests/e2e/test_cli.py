"""End-to-end tests for the EchoVector CLI."""

from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from echovector.cli.main import app

runner = CliRunner()


def _write_tone(path: Path, frequency: float = 440.0) -> None:
    sample_rate = 16_000
    duration = 0.25
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sf.write(path, 0.25 * np.sin(2 * np.pi * frequency * t), sample_rate)


def _index(store_dir: Path, *args: str):
    return runner.invoke(
        app, ["index", *args, "--store-dir", str(store_dir), "--backend", "local"]
    )


def test_index_search_and_stats_commands(tmp_path: Path) -> None:
    """Test the CLI against a real local index."""
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    _write_tone(audio_path)

    index_result = _index(store_dir, str(audio_path))
    assert index_result.exit_code == 0
    assert "Indexing complete." in index_result.stdout
    assert "Indexed 1 chunk(s)." in index_result.stdout

    search_result = runner.invoke(
        app,
        [
            "search",
            "high tone",
            "--top-k",
            "1",
            "--store-dir",
            str(store_dir),
            "--backend",
            "local",
        ],
    )
    assert search_result.exit_code == 0
    assert "Top 1 Results" in search_result.stdout
    assert "tone.wav" in search_result.stdout

    stats_result = runner.invoke(
        app,
        ["stats", "--store-dir", str(store_dir), "--backend", "local"],
    )
    assert stats_result.exit_code == 0
    assert "Index Statistics" in stats_result.stdout
    assert "chunks" in stats_result.stdout
    assert "files" in stats_result.stdout


def test_stats_reports_the_backend_that_built_the_index(tmp_path: Path) -> None:
    """The manifest is what makes a store self-describing."""
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    _write_tone(audio_path)
    _index(store_dir, str(audio_path))

    result = runner.invoke(app, ["stats", "--store-dir", str(store_dir), "--backend", "local"])

    assert result.exit_code == 0
    assert "local:acoustic-v2" in result.stdout.replace("\n", "")


def test_device_flag_is_ignored_by_cpu_only_backends(tmp_path: Path) -> None:
    """--device is generic; a backend that ignores it must not make the run fail."""
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    _write_tone(audio_path)

    result = _index(store_dir, str(audio_path), "--device", "cpu")

    assert result.exit_code == 0
    assert "does not use --device" in result.stdout
    assert "Indexed 1 chunk(s)." in result.stdout


def test_reset_clears_the_store(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    _write_tone(audio_path)

    assert _index(store_dir, str(audio_path)).exit_code == 0
    result = _index(store_dir, str(audio_path), "--reset")

    assert result.exit_code == 0
    assert "Indexed 1 chunk(s)." in result.stdout

    stats = runner.invoke(app, ["stats", "--store-dir", str(store_dir), "--backend", "local"])
    assert "│ chunks" in stats.stdout


def test_index_reports_when_nothing_is_new(tmp_path: Path) -> None:
    """A second run over the same files should explain the zero, not just print it."""
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    _write_tone(audio_path)
    _index(store_dir, str(audio_path))

    result = _index(store_dir, str(audio_path))

    assert result.exit_code == 0
    assert "Indexed 0 chunk(s)." in result.stdout
    assert "--force" in result.stdout


def test_force_reindexes_without_growing_the_index(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    _write_tone(audio_path)
    _index(store_dir, str(audio_path))

    result = _index(store_dir, str(audio_path), "--force")

    assert result.exit_code == 0
    assert "Indexed 1 chunk(s)." in result.stdout


def test_no_recursive_skips_nested_directories(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    (audio_dir / "nested").mkdir(parents=True)
    _write_tone(audio_dir / "top.wav")
    _write_tone(audio_dir / "nested" / "deep.wav")
    store_dir = tmp_path / "index"

    result = _index(store_dir, str(audio_dir), "--no-recursive")

    assert result.exit_code == 0
    assert "Indexed 1 chunk(s)." in result.stdout


def test_recursive_is_the_default(tmp_path: Path) -> None:
    """The CLI default must match the Python API default."""
    audio_dir = tmp_path / "audio"
    (audio_dir / "nested").mkdir(parents=True)
    _write_tone(audio_dir / "top.wav")
    _write_tone(audio_dir / "nested" / "deep.wav")
    store_dir = tmp_path / "index"

    result = _index(store_dir, str(audio_dir))

    assert result.exit_code == 0
    assert "Indexed 2 chunk(s)." in result.stdout


def test_search_min_score_filters_results(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    _write_tone(audio_path)
    _index(store_dir, str(audio_path))

    result = runner.invoke(
        app,
        [
            "search",
            "high tone",
            "--store-dir",
            str(store_dir),
            "--backend",
            "local",
            "--min-score",
            "1.5",
        ],
    )

    assert result.exit_code == 0
    assert "No results found." in result.stdout


def test_search_without_an_index_says_so(tmp_path: Path) -> None:
    """A missing index and a genuine miss must not look identical."""
    store_dir = tmp_path / "index"
    result = runner.invoke(
        app,
        ["search", "anything", "--store-dir", str(store_dir), "--backend", "local"],
    )
    assert result.exit_code == 1
    assert "No index found" in result.stdout


def test_stats_without_an_index_says_so(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["stats", "--store-dir", str(tmp_path / "index"), "--backend", "local"],
    )
    assert result.exit_code == 1
    assert "No index found" in result.stdout


def test_search_with_unknown_backend_reports_error(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    _write_tone(audio_path)
    _index(store_dir, str(audio_path))

    result = runner.invoke(
        app,
        ["search", "anything", "--store-dir", str(store_dir), "--backend", "nonexistent"],
    )
    assert result.exit_code == 1
    assert "Search failed" in result.stdout


def test_index_with_unknown_backend_reports_error(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path)
    result = runner.invoke(
        app,
        [
            "index",
            str(audio_path),
            "--store-dir",
            str(tmp_path / "index"),
            "--backend",
            "nonexistent",
        ],
    )
    assert result.exit_code == 1
    assert "Indexing failed" in result.stdout


def test_searching_with_the_wrong_backend_is_refused(tmp_path: Path) -> None:
    """Scores from a different embedding space are meaningless, not merely worse."""
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    _write_tone(audio_path)
    _index(store_dir, str(audio_path))

    result = runner.invoke(
        app,
        ["search", "anything", "--store-dir", str(store_dir), "--backend", "whisper"],
    )
    assert result.exit_code == 1
    assert "Search failed" in result.stdout
    assert "re-index" in result.stdout.replace("\n", " ").lower()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_help_command() -> None:
    """Test the help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "EchoVector CLI" in result.stdout
    assert "index" in result.stdout
    assert "search" in result.stdout
    assert "stats" in result.stdout
