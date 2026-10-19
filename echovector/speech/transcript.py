"""Word-level transcripts and the overlapping chunks that get embedded."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Word:
    """One transcribed word with its position in the audio, in seconds."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Chunk:
    """A window of consecutive words, the unit that gets embedded and returned."""

    text: str
    start: float
    end: float


def chunk_words(words: list[Word], size: int = 45, stride: int = 25) -> list[Chunk]:
    """Overlapping word windows, since whisper segments are too short to hold a topic."""
    usable = [w for w in words if w.text.strip()]
    chunks: list[Chunk] = []
    for begin in range(0, max(len(usable) - size + 1, 1), stride):
        window = usable[begin : begin + size]
        if not window:
            break
        chunks.append(
            Chunk(
                text=" ".join(w.text.strip() for w in window),
                start=window[0].start,
                end=window[-1].end,
            )
        )
    return chunks


def save_transcript(path: Path, words: list[Word]) -> None:
    """Write a transcript so the one expensive transcription pass is never repeated."""
    path.write_text(json.dumps([[w.text, w.start, w.end] for w in words]))


def load_transcript(path: Path) -> list[Word]:
    """Read a transcript written by `save_transcript`."""
    return [Word(text, start, end) for text, start, end in json.loads(path.read_text())]
