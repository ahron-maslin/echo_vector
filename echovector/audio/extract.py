"""Pull the audio behind a search hit out of its source file."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

from echovector.audio import dsp

DEFAULT_SUBTYPE = "FLOAT"
"""Subtype used when writing clips.

soundfile silently narrows float input to PCM_16, which would quantize the clip
relative to the audio the index actually saw.
"""

SUPPORTED_PLAYERS: tuple[str, ...] = ("ffplay", "mpv", "vlc")
"""Players ``player_command`` knows how to build a seek invocation for."""


def extract_clip(
    filepath: str | Path,
    start: float,
    end: float,
    out_path: str | Path | None = None,
    padding: float = 0.0,
    sample_rate: int | None = None,
    subtype: str | None = None,
) -> Path | npt.NDArray[np.float32]:
    """Extract the audio between two timestamps without decoding the whole file.

    Only the requested frames are read: the decoder seeks to ``start`` and reads a
    bounded number of frames, so pulling ten seconds out of a three-hour recording
    costs ten seconds of memory.

    Args:
        filepath: Path to the source audio file.
        start: Start of the range, in seconds from the beginning of the file.
        end: End of the range, in seconds. Must be greater than ``start``.
        out_path: Where to write the clip. When omitted the waveform is returned
            instead of being written.
        padding: Extra seconds of context to include on both sides, clamped to the
            bounds of the file.
        sample_rate: Resample the clip to this rate. Defaults to keeping the source
            rate.
        subtype: soundfile subtype for the written file, e.g. ``"PCM_16"``. Defaults
            to ``FLOAT`` where the output container supports it.

    Returns:
        ``out_path`` as a :class:`~pathlib.Path` when writing, otherwise the clip as
        a float32 array shaped ``(frames,)`` for mono or ``(frames, channels)``.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
        ValueError: If ``start`` is negative, ``end`` is not after ``start``,
            ``padding`` is negative, ``sample_rate`` is not positive, the requested
            range lies entirely past the end of the file, or an explicitly requested
            ``subtype`` is not valid for the output container.
    """
    source = Path(filepath)
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}.")
    if end <= start:
        raise ValueError(f"end must be greater than start, got start={start}, end={end}.")
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}.")
    if sample_rate is not None and sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")

    with sf.SoundFile(str(source)) as handle:
        source_rate = int(handle.samplerate)
        total_frames = int(handle.frames)
        duration = total_frames / source_rate

        padded_start = max(0.0, start - padding)
        if padded_start >= duration:
            raise ValueError(
                f"Requested range starts at {padded_start:.3f}s but {source} is only "
                f"{duration:.3f}s long."
            )

        # A range that runs off the end is clamped rather than rejected: a hit near
        # the tail of a file is still a hit.
        padded_end = min(duration, end + padding)
        start_frame = min(round(padded_start * source_rate), total_frames)
        end_frame = min(round(padded_end * source_rate), total_frames)
        if end_frame <= start_frame:
            raise ValueError(
                f"Requested range {start}s-{end}s is empty at {source_rate} Hz in {source}."
            )

        handle.seek(start_frame)
        clip = np.asarray(
            handle.read(end_frame - start_frame, dtype="float32", always_2d=False),
            dtype=np.float32,
        )

    out_rate = source_rate
    if sample_rate is not None and sample_rate != source_rate:
        clip = np.asarray(
            dsp.resample(clip, source_rate, sample_rate),
            dtype=np.float32,
        )
        out_rate = sample_rate

    if out_path is None:
        return clip

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(destination),
        clip,
        samplerate=out_rate,
        subtype=_resolve_subtype(destination, subtype),
    )
    return destination


def player_command(
    filepath: str | Path,
    start: float,
    end: float,
    player: str = "ffplay",
) -> list[str]:
    """Build the argv for playing one range of an audio file.

    Nothing is executed; the caller decides whether and how to run the command.

    Args:
        filepath: Path to the audio file to play.
        start: Start of the range, in seconds.
        end: End of the range, in seconds. Must be greater than ``start``.
        player: Player to build the command for. One of :data:`SUPPORTED_PLAYERS`.

    Returns:
        The argv list, with the executable name first.

    Raises:
        ValueError: If ``start`` is negative, ``end`` is not after ``start``, or
            ``player`` is not supported.
    """
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}.")
    if end <= start:
        raise ValueError(f"end must be greater than start, got start={start}, end={end}.")

    name = player.strip().lower()
    if name not in SUPPORTED_PLAYERS:
        supported = ", ".join(SUPPORTED_PLAYERS)
        raise ValueError(f"Unsupported player {player!r}. Supported players: {supported}.")

    target = str(filepath)
    begin = f"{start:.3f}"
    duration = f"{end - start:.3f}"

    if name == "ffplay":
        return ["ffplay", "-nodisp", "-autoexit", "-ss", begin, "-t", duration, target]
    if name == "mpv":
        return ["mpv", f"--start={begin}", f"--length={duration}", target]
    return ["vlc", f"--start-time={begin}", f"--stop-time={end:.3f}", target, "vlc://quit"]


def _resolve_subtype(out_path: Path, subtype: str | None) -> str | None:
    """Pick the soundfile subtype to write ``out_path`` with.

    Args:
        out_path: Destination path; its extension selects the container format.
        subtype: Explicitly requested subtype, or ``None`` to prefer ``FLOAT``.

    Returns:
        The subtype to pass to :func:`soundfile.write`, or ``None`` to let libsndfile
        choose the container default.

    Raises:
        ValueError: If an explicitly requested ``subtype`` is not valid for the
            container implied by ``out_path``.
    """
    container = out_path.suffix.lstrip(".").upper()
    requested = DEFAULT_SUBTYPE if subtype is None else subtype.upper()

    if bool(sf.check_format(container, requested)):
        return requested
    if subtype is not None:
        raise ValueError(f"Subtype {subtype!r} is not supported for {container or 'this'} files.")

    # Compressed containers (mp3, ogg) have no float subtype; their own default is
    # the best available fidelity.
    return None
