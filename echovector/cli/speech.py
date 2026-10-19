"""`echovector speech`: concept search over spoken recordings (transcribe, embed, search)."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from echovector.speech.backends import DEFAULT_EMBEDDER, DEFAULT_WHISPER

if TYPE_CHECKING:
    from echovector.speech import SpeechIndex

speech_app = typer.Typer(help="Concept search over recorded speech.", add_completion=False)
console = Console()

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

SpeechDirOption = Annotated[
    Path, typer.Option("--store-dir", help="Directory for the speech index.")
]


def _open(store_dir: Path, embedder: str) -> "SpeechIndex":
    # The speech stack is an optional extra, so it is imported only when a speech command runs.
    try:
        from echovector.speech import SpeechIndex
        from echovector.speech.backends import FastEmbedder, espeak_ipa
    except ImportError as exc:
        console.print(
            "[red]Speech search needs the extra: pip install 'echo_vector[speech]'[/red]"
        )
        raise typer.Exit(code=1) from exc
    return SpeechIndex(store_dir, embedder=FastEmbedder(embedder), ipa=espeak_ipa)


@speech_app.command("index")
def index_speech(
    path: Annotated[Path, typer.Argument(help="Audio file or folder of audio files.")],
    store_dir: SpeechDirOption = Path(".echovector-speech"),
    whisper_model: Annotated[str, typer.Option("--whisper-model")] = DEFAULT_WHISPER,
    embedder: Annotated[str, typer.Option("--embedder")] = DEFAULT_EMBEDDER,
) -> None:
    """Transcribe audio once (cached beside the index), then embed it for search."""
    files = (
        sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
        if path.is_dir()
        else [path]
    )
    if not files:
        console.print(f"[red]No audio files under {path}[/red]")
        raise typer.Exit(code=1)

    speech = _open(store_dir, embedder)
    from echovector.speech import load_transcript, save_transcript
    from echovector.speech.backends import transcribe

    cache = store_dir / "transcripts"
    cache.mkdir(parents=True, exist_ok=True)
    for audio in files:
        transcript = cache / f"{audio.stem}.json"
        if transcript.exists():
            words = load_transcript(transcript)
        else:
            console.print(f"Transcribing {audio.name}")
            words = transcribe(audio, whisper_model)
            save_transcript(transcript, words)
        speech.add(str(audio), words)
        speech.save()
    console.print(f"[green]Indexed {len(files)} file(s) into {store_dir}[/green]")


@speech_app.command("search")
def search_speech(
    query: Annotated[str, typer.Argument(help="Concept or term to look for.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 5,
    store_dir: SpeechDirOption = Path(".echovector-speech"),
    embedder: Annotated[str, typer.Option("--embedder")] = DEFAULT_EMBEDDER,
) -> None:
    """Find passages by meaning, absorbing whisper's inconsistent spelling of Hebrew terms."""
    if not (store_dir / "chunks.json").exists():
        console.print(f"[red]No speech index at {store_dir}. Run `echovector speech index`.[/red]")
        raise typer.Exit(code=1)

    hits = _open(store_dir, embedder).search(query, k=top_k)
    table = Table(title=f"Top {len(hits)} for '{query}'")
    for column in ("Score", "File", "At", "Passage"):
        table.add_column(column)
    for hit in hits:
        minutes, seconds = divmod(int(hit.start), 60)
        table.add_row(
            f"{hit.score:.3f}", Path(hit.file).name, f"{minutes}:{seconds:02d}", hit.text[:160]
        )
    console.print(table)
