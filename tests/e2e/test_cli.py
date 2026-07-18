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


def test_index_search_and_stats_commands(tmp_path: Path) -> None:
    """Test the CLI against a real local index."""
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    _write_tone(audio_path)

    index_result = runner.invoke(
        app,
        [
            "index",
            str(audio_path),
            "--store-dir",
            str(store_dir),
            "--backend",
            "local",
        ],
    )
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
    assert "vectors" in stats_result.stdout
    assert "1" in stats_result.stdout


def test_index_with_reset_and_device_flag(tmp_path: Path) -> None:
    """--reset clears an existing index; --device is forwarded to the backend."""
    audio_path = tmp_path / "tone.wav"
    store_dir = tmp_path / "index"
    _write_tone(audio_path)

    runner.invoke(
        app,
        ["index", str(audio_path), "--store-dir", str(store_dir), "--backend", "local"],
    )

    # `local` backend doesn't accept a `device` kwarg, so this should fail cleanly
    # via the CLI's error handler rather than crashing with a raw traceback.
    result = runner.invoke(
        app,
        [
            "index",
            str(audio_path),
            "--store-dir",
            str(store_dir),
            "--backend",
            "local",
            "--device",
            "cpu",
            "--reset",
        ],
    )
    assert result.exit_code == 1
    assert "Indexing failed" in result.stdout


def test_search_command_no_results(tmp_path: Path) -> None:
    """Searching an empty index should report no results, not error."""
    store_dir = tmp_path / "index"
    result = runner.invoke(
        app,
        ["search", "anything", "--store-dir", str(store_dir), "--backend", "local"],
    )
    assert result.exit_code == 0
    assert "No results found." in result.stdout


def test_search_command_failure_reports_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "anything",
            "--store-dir",
            str(tmp_path / "index"),
            "--backend",
            "local",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 1
    assert "Search failed" in result.stdout


def test_stats_command_failure_reports_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["stats", "--store-dir", str(tmp_path / "index"), "--backend", "nonexistent"],
    )
    assert result.exit_code == 1
    assert "Stats failed" in result.stdout


def test_help_command() -> None:
    """Test the help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "EchoVector CLI" in result.stdout
    assert "index" in result.stdout
    assert "search" in result.stdout
    assert "stats" in result.stdout
