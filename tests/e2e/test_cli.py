"""End-to-end tests for the EchoVector CLI."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
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


def test_offered_backends_are_all_implemented(tmp_path: Path) -> None:
    """The backend list must not advertise names that fail once you type them.

    `whisper` was offered for a long time and raised NotImplementedError only
    after the user had chosen it.
    """
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
            "whisper",
        ],
    )
    offered = result.stdout.replace("\n", " ")

    assert result.exit_code == 1
    assert "whisper" not in offered.split("Valid options are:")[-1]
    for implemented in ("clap", "local", "wav2vec2", "hubert", "ast"):
        assert implemented in offered


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


# ---------------------------------------------------------------------------
# similar / extract / play
# ---------------------------------------------------------------------------


def test_similar_finds_matching_audio(tmp_path: Path) -> None:
    """Audio-to-audio search needs no text tower at all."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_tone(audio_dir / "query.wav", frequency=880.0)
    _write_tone(audio_dir / "close.wav", frequency=900.0)
    _write_tone(audio_dir / "far.wav", frequency=80.0)
    store_dir = tmp_path / "index"
    _index(store_dir, str(audio_dir))

    result = runner.invoke(
        app,
        [
            "similar",
            str(audio_dir / "query.wav"),
            "--store-dir",
            str(store_dir),
            "--backend",
            "local",
            "--top-k",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "close.wav" in result.stdout
    assert "query.wav" not in result.stdout.split("Timestamp")[-1]


def test_similar_can_include_the_source_file(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_tone(audio_dir / "query.wav")
    store_dir = tmp_path / "index"
    _index(store_dir, str(audio_dir))

    result = runner.invoke(
        app,
        [
            "similar",
            str(audio_dir / "query.wav"),
            "--store-dir",
            str(store_dir),
            "--backend",
            "local",
            "--include-source",
        ],
    )

    assert result.exit_code == 0
    assert "query.wav" in result.stdout


def test_similar_without_an_index_says_so(tmp_path: Path) -> None:
    _write_tone(tmp_path / "q.wav")
    result = runner.invoke(
        app,
        ["similar", str(tmp_path / "q.wav"), "--store-dir", str(tmp_path / "nope")],
    )
    assert result.exit_code == 1
    assert "No index found" in result.stdout


def test_similar_reports_a_missing_query_file(tmp_path: Path) -> None:
    store_dir = tmp_path / "index"
    _write_tone(tmp_path / "a.wav")
    _index(store_dir, str(tmp_path / "a.wav"))

    result = runner.invoke(
        app,
        [
            "similar",
            str(tmp_path / "missing.wav"),
            "--store-dir",
            str(store_dir),
            "--backend",
            "local",
        ],
    )
    assert result.exit_code == 1
    assert "Similarity search failed" in result.stdout


def test_extract_writes_the_clip(tmp_path: Path) -> None:
    """A hit is only useful if you can actually hear it."""
    import soundfile as sf

    source = tmp_path / "tone.wav"
    _write_tone(source)
    out = tmp_path / "clip.wav"

    result = runner.invoke(app, ["extract", str(source), "0.05", "0.15", "--out", str(out)])

    assert result.exit_code == 0
    assert out.exists()
    assert sf.info(out).duration == pytest.approx(0.1, abs=0.01)


def test_extract_accepts_padding(tmp_path: Path) -> None:
    import soundfile as sf

    source = tmp_path / "tone.wav"
    _write_tone(source)
    out = tmp_path / "clip.wav"

    result = runner.invoke(
        app,
        ["extract", str(source), "0.10", "0.15", "--out", str(out), "--padding", "0.02"],
    )

    assert result.exit_code == 0
    assert sf.info(out).duration == pytest.approx(0.09, abs=0.01)


def test_extract_reports_an_invalid_range(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_tone(source)

    result = runner.invoke(
        app,
        ["extract", str(source), "0.2", "0.1", "--out", str(tmp_path / "clip.wav")],
    )

    assert result.exit_code == 1
    assert "Extract failed" in result.stdout


def test_play_prints_a_command_without_running_it(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_tone(source)

    result = runner.invoke(app, ["play", str(source), "12.0", "20.0"])

    assert result.exit_code == 0
    assert "ffplay" in result.stdout
    assert "12" in result.stdout


def test_play_rejects_an_unknown_player(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_tone(source)

    result = runner.invoke(app, ["play", str(source), "0.0", "1.0", "--player", "winamp"])

    assert result.exit_code == 1
    assert "Cannot build a play command" in result.stdout


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def _golden_set(path: Path, audio: Path) -> Path:
    payload = {
        "name": "cli-smoke",
        "queries": [
            {
                "id": "high",
                "query": "high bright alarm tone",
                "relevant": [{"filepath": str(audio.resolve()), "start": 0.0, "end": 0.25}],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_eval_reports_retrieval_quality(tmp_path: Path) -> None:
    """Without this command, a tuning change is a guess."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "high.wav"
    _write_tone(audio, frequency=880.0)
    store_dir = tmp_path / "index"
    _index(store_dir, str(audio_dir))
    golden = _golden_set(tmp_path / "golden.json", audio)

    result = runner.invoke(
        app,
        [
            "eval",
            str(golden),
            "--store-dir",
            str(store_dir),
            "--backend",
            "local",
            "--ks",
            "1",
            "--per-query",
        ],
    )

    assert result.exit_code == 0
    assert "recall" in result.stdout
    assert "ndcg" in result.stdout


def test_eval_emits_json(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "high.wav"
    _write_tone(audio, frequency=880.0)
    store_dir = tmp_path / "index"
    _index(store_dir, str(audio_dir))
    golden = _golden_set(tmp_path / "golden.json", audio)

    result = runner.invoke(
        app,
        [
            "eval",
            str(golden),
            "--store-dir",
            str(store_dir),
            "--backend",
            "local",
            "--ks",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "aggregates" in result.stdout


def test_eval_rejects_empty_cutoffs(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "high.wav"
    _write_tone(audio)
    store_dir = tmp_path / "index"
    _index(store_dir, str(audio_dir))
    golden = _golden_set(tmp_path / "golden.json", audio)

    result = runner.invoke(
        app,
        [
            "eval",
            str(golden),
            "--store-dir",
            str(store_dir),
            "--backend",
            "local",
            "--ks",
            " ",
        ],
    )

    assert result.exit_code == 1
    assert "Evaluation failed" in result.stdout


def test_eval_reports_a_malformed_golden_set(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_tone(audio_dir / "high.wav")
    store_dir = tmp_path / "index"
    _index(store_dir, str(audio_dir))
    bad = tmp_path / "bad.json"
    bad.write_text('{"queries": []}', encoding="utf-8")

    result = runner.invoke(
        app,
        ["eval", str(bad), "--store-dir", str(store_dir), "--backend", "local"],
    )

    assert result.exit_code == 1
    assert "Evaluation failed" in result.stdout


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def test_prune_removes_deleted_files(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_tone(audio_dir / "keep.wav", frequency=440.0)
    _write_tone(audio_dir / "gone.wav", frequency=880.0)
    store_dir = tmp_path / "index"
    _index(store_dir, str(audio_dir))

    (audio_dir / "gone.wav").unlink()
    result = runner.invoke(
        app,
        ["prune", "--store-dir", str(store_dir), "--backend", "local"],
    )

    assert result.exit_code == 0
    assert "Removed 1 deleted file(s)." in result.stdout
    assert "gone.wav" in result.stdout


def test_prune_check_does_not_modify_the_index(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_tone(audio_dir / "gone.wav")
    store_dir = tmp_path / "index"
    _index(store_dir, str(audio_dir))
    (audio_dir / "gone.wav").unlink()

    check = runner.invoke(
        app, ["prune", "--store-dir", str(store_dir), "--backend", "local", "--check"]
    )
    after = runner.invoke(app, ["stats", "--store-dir", str(store_dir), "--backend", "local"])

    assert check.exit_code == 0
    assert "Would remove 1 deleted file(s)." in check.stdout
    assert "│ files" in after.stdout
    assert "1" in after.stdout


def test_prune_reports_changed_files(tmp_path: Path) -> None:
    """An edited recording is reported so the user knows to re-index it."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "tone.wav"
    _write_tone(audio, frequency=440.0)
    store_dir = tmp_path / "index"
    _index(store_dir, str(audio_dir))

    _write_tone(audio, frequency=880.0)
    os.utime(audio, (0, 0))
    result = runner.invoke(app, ["prune", "--store-dir", str(store_dir), "--backend", "local"])

    assert result.exit_code == 0
    assert "1 file(s) changed since indexing" in result.stdout
    assert "tone.wav" in result.stdout


def test_prune_without_an_index_says_so(tmp_path: Path) -> None:
    result = runner.invoke(app, ["prune", "--store-dir", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "No index found" in result.stdout
