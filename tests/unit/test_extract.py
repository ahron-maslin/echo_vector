from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import soundfile as sf

from echovector.audio import extract as extract_module
from echovector.audio.extract import _resolve_subtype, extract_clip, player_command

SR = 8000


def _ramp(seconds: float, sample_rate: int = SR) -> np.ndarray:
    """Signal whose value encodes its own sample index, so slices are identifiable."""
    n = int(seconds * sample_rate)
    return (np.arange(n, dtype=np.float32) / max(n, 1)).astype(np.float32)


def _write(path: Path, audio: np.ndarray, sample_rate: int = SR) -> Path:
    sf.write(str(path), audio, sample_rate, subtype="FLOAT")
    return path


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return _write(tmp_path / "source.wav", _ramp(10.0))


class SpySoundFile(sf.SoundFile):
    """Records every read so tests can prove the whole file is never decoded."""

    reads: ClassVar[list[int]] = []
    seeks: ClassVar[list[int]] = []

    def seek(self, frames, whence=0):  # type: ignore[no-untyped-def]
        type(self).seeks.append(frames)
        return super().seek(frames, whence)

    def read(self, frames=-1, **kwargs):  # type: ignore[no-untyped-def]
        type(self).reads.append(frames)
        return super().read(frames, **kwargs)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> type[SpySoundFile]:
    SpySoundFile.reads = []
    SpySoundFile.seeks = []
    monkeypatch.setattr(extract_module.sf, "SoundFile", SpySoundFile)
    return SpySoundFile


# --------------------------------------------------------------------------
# extract_clip
# --------------------------------------------------------------------------


def test_returns_waveform_when_no_out_path(source: Path) -> None:
    clip = extract_clip(source, 2.0, 4.0)

    assert isinstance(clip, np.ndarray)
    assert clip.dtype == np.float32
    assert len(clip) == 2 * SR
    np.testing.assert_allclose(clip, _ramp(10.0)[2 * SR : 4 * SR])


def test_writes_file_when_out_path_given(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "clip.wav"

    returned = extract_clip(source, 1.0, 3.5, out_path=out)

    assert returned == out
    written, rate = sf.read(str(out), dtype="float32")
    assert rate == SR
    assert len(written) == int(2.5 * SR)
    np.testing.assert_allclose(written, _ramp(10.0)[SR : int(3.5 * SR)])


def test_written_clip_is_float_not_quantized(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "clip.wav"

    extract_clip(source, 0.0, 1.0, out_path=out)

    assert sf.info(str(out)).subtype == "FLOAT"


def test_explicit_subtype_is_honoured(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "clip.wav"

    extract_clip(source, 0.0, 1.0, out_path=out, subtype="PCM_16")

    assert sf.info(str(out)).subtype == "PCM_16"


def test_explicit_subtype_invalid_for_container(source: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not supported"):
        extract_clip(source, 0.0, 1.0, out_path=tmp_path / "clip.wav", subtype="VORBIS")


@pytest.mark.parametrize("suffix", [".ogg", ".mp3", ""])
def test_float_default_falls_back_for_containers_without_float(suffix: str) -> None:
    assert _resolve_subtype(Path(f"clip{suffix}"), None) is None


def test_out_path_parent_is_created(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deeper" / "clip.wav"

    extract_clip(source, 0.0, 0.5, out_path=out)

    assert out.exists()


def test_padding_widens_both_sides(source: Path) -> None:
    clip = extract_clip(source, 4.0, 5.0, padding=1.5)

    assert isinstance(clip, np.ndarray)
    np.testing.assert_allclose(clip, _ramp(10.0)[int(2.5 * SR) : int(6.5 * SR)])


def test_padding_is_clamped_to_file_bounds(source: Path) -> None:
    clip = extract_clip(source, 0.5, 9.5, padding=30.0)

    assert isinstance(clip, np.ndarray)
    assert len(clip) == 10 * SR


def test_range_partially_past_eof_is_clamped(source: Path) -> None:
    clip = extract_clip(source, 9.0, 25.0)

    assert isinstance(clip, np.ndarray)
    assert len(clip) == SR


def test_source_sample_rate_is_preserved(tmp_path: Path) -> None:
    src = _write(tmp_path / "odd.wav", _ramp(3.0, sample_rate=22050), sample_rate=22050)
    out = tmp_path / "clip.wav"

    extract_clip(src, 0.0, 1.0, out_path=out)

    assert sf.info(str(out)).samplerate == 22050


def test_resamples_when_sample_rate_requested(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "clip.wav"

    extract_clip(source, 1.0, 3.0, out_path=out, sample_rate=16000)

    info = sf.info(str(out))
    assert info.samplerate == 16000
    assert info.frames == pytest.approx(2 * 16000, abs=8)


def test_stereo_channels_are_preserved(tmp_path: Path) -> None:
    stereo = np.stack([_ramp(4.0), _ramp(4.0) * -1.0], axis=1)
    src = _write(tmp_path / "stereo.wav", stereo)

    clip = extract_clip(src, 1.0, 2.0)

    assert isinstance(clip, np.ndarray)
    assert clip.shape == (SR, 2)


# --------------------------------------------------------------------------
# Bounded reads
# --------------------------------------------------------------------------


def test_only_the_requested_frames_are_read(source: Path, spy: type[SpySoundFile]) -> None:
    extract_clip(source, 2.0, 4.0)

    # soundfile issues its own relative seeks inside read(), so only the first
    # seek is ours.
    assert spy.seeks[0] == 2 * SR
    assert spy.reads == [2 * SR]


def test_long_file_is_not_fully_decoded(tmp_path: Path, spy: type[SpySoundFile]) -> None:
    # 30 minutes at 8 kHz; extracting 10s must touch only 10s worth of frames.
    long_file = _write(tmp_path / "podcast.wav", np.zeros(1800 * SR, dtype=np.float32))
    total_frames = sf.info(str(long_file)).frames

    clip = extract_clip(long_file, 1500.0, 1510.0)

    assert isinstance(clip, np.ndarray)
    assert len(clip) == 10 * SR
    assert sum(spy.reads) == 10 * SR
    assert sum(spy.reads) < total_frames / 100
    assert -1 not in spy.reads


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        extract_clip(tmp_path / "nope.wav", 0.0, 1.0)


def test_negative_start(source: Path) -> None:
    with pytest.raises(ValueError, match="start must be non-negative"):
        extract_clip(source, -1.0, 5.0)


@pytest.mark.parametrize(("start", "end"), [(5.0, 5.0), (5.0, 4.0)])
def test_end_not_after_start(source: Path, start: float, end: float) -> None:
    with pytest.raises(ValueError, match="end must be greater than start"):
        extract_clip(source, start, end)


def test_negative_padding(source: Path) -> None:
    with pytest.raises(ValueError, match="padding must be non-negative"):
        extract_clip(source, 1.0, 2.0, padding=-0.5)


def test_non_positive_sample_rate(source: Path) -> None:
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        extract_clip(source, 1.0, 2.0, sample_rate=0)


def test_range_entirely_past_eof(source: Path) -> None:
    with pytest.raises(ValueError, match=r"only 10\.000s long"):
        extract_clip(source, 12.0, 14.0)


def test_range_shorter_than_one_frame(source: Path) -> None:
    with pytest.raises(ValueError, match="is empty at"):
        extract_clip(source, 1.0, 1.00001)


def test_empty_source_file(tmp_path: Path) -> None:
    empty = _write(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32))

    with pytest.raises(ValueError, match=r"only 0\.000s long"):
        extract_clip(empty, 0.0, 1.0)


# --------------------------------------------------------------------------
# player_command
# --------------------------------------------------------------------------


def test_ffplay_command() -> None:
    assert player_command("/a/b.wav", 124.0, 134.0) == [
        "ffplay",
        "-nodisp",
        "-autoexit",
        "-ss",
        "124.000",
        "-t",
        "10.000",
        "/a/b.wav",
    ]


def test_mpv_command() -> None:
    assert player_command("/a/b.wav", 5.5, 8.0, player="mpv") == [
        "mpv",
        "--start=5.500",
        "--length=2.500",
        "/a/b.wav",
    ]


def test_vlc_command_uses_stop_time() -> None:
    assert player_command(Path("/a/b.wav"), 5.5, 8.0, player="vlc") == [
        "vlc",
        "--start-time=5.500",
        "--stop-time=8.000",
        "/a/b.wav",
        "vlc://quit",
    ]


def test_player_name_is_case_insensitive() -> None:
    assert player_command("/a/b.wav", 0.0, 1.0, player=" MPV ")[0] == "mpv"


def test_unknown_player() -> None:
    with pytest.raises(ValueError, match="Unsupported player"):
        player_command("/a/b.wav", 0.0, 1.0, player="winamp")


@pytest.mark.parametrize(("start", "end"), [(-1.0, 5.0), (5.0, 5.0)])
def test_player_command_validates_range(start: float, end: float) -> None:
    with pytest.raises(ValueError):
        player_command("/a/b.wav", start, end)
