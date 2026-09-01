"""Command-line interface for EchoVector."""

import inspect
import json
import shlex
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress
from rich.table import Table

from echovector import EchoVector, SearchFilter, __version__
from echovector.audio.extract import SUPPORTED_PLAYERS, extract_clip, player_command
from echovector.core import DEFAULT_CHUNK_SECONDS
from echovector.embeddings.factory import backend_class_for
from echovector.evaluation import RelevanceJudge, evaluate, load_golden_set
from echovector.indexing.manifest import IndexManifest
from echovector.search.results import SearchResult
from echovector.watch import DEFAULT_INTERVAL_SECONDS, FolderWatcher, WatchEvent

app = typer.Typer(
    name="echovector",
    help="EchoVector CLI for semantic text search over audio files.",
    add_completion=False,
)
console = Console()

StoreDirOption = Annotated[Path, typer.Option("--store-dir", help="Directory for index files.")]
BackendOption = Annotated[str, typer.Option("--backend", help="Embedding backend to use.")]
DeviceOption = Annotated[
    str | None,
    typer.Option("--device", help="Model device, e.g. cpu or cuda. Ignored by CPU-only backends."),
]


def _fail(what: str, exc: Exception) -> None:
    """Print an error without letting Rich eat parts of the message.

    Exception text is data, not markup. `pip install 'echo_vector[clap]'` would
    otherwise render as `pip install 'echo_vector'` because Rich reads `[clap]`
    as a style tag -- silently destroying the one thing the message exists to
    convey.
    """
    console.print(f"[bold red]{what}:[/bold red] {escape(str(exc))}")


def _backend_kwargs(backend: str, device: str | None) -> dict[str, Any]:
    """Build backend kwargs, dropping options the backend cannot accept.

    `--device` is a generic flag, but backends like `local` run on CPU only and
    take no `device` argument. Forwarding it blindly turned a meaningless flag
    into a hard failure.
    """
    if device is None:
        return {}
    try:
        parameters = inspect.signature(backend_class_for(backend).__init__).parameters
    except (ValueError, TypeError):  # pragma: no cover - builtins without signatures
        return {"device": device}
    if "device" in parameters:
        return {"device": device}
    console.print(
        f"[yellow]Note:[/yellow] the '{backend}' backend does not use --device; ignoring it."
    )
    return {}


@app.command()
def index(
    files: Annotated[
        list[Path],
        typer.Argument(help="Audio files or directories to index."),
    ],
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive",
            "-r",
            help="Search directories recursively for audio files.",
        ),
    ] = True,
    store_dir: StoreDirOption = Path(".echovector"),
    backend: BackendOption = "clap",
    device: DeviceOption = None,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Clear the existing index before indexing."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-index files that are already in the store."),
    ] = False,
    chunk_seconds: Annotated[
        float,
        typer.Option("--chunk-seconds", help="Audio chunk size to embed."),
    ] = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: Annotated[
        float,
        typer.Option("--overlap-seconds", help="Overlap between adjacent chunks."),
    ] = 1.0,
) -> None:
    """Index audio files for search."""
    try:
        engine = EchoVector(
            store_dir=store_dir,
            backend=backend,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            **_backend_kwargs(backend, device),
        )
        if reset:
            engine.reset()

        with Progress(console=console) as progress:
            task = progress.add_task("Indexing files...", total=None)

            def _on_file_indexed(done: int, total: int, file_path: Path) -> None:
                progress.update(
                    task, total=total, completed=done, description=f"Indexed {file_path.name}"
                )

            count = engine.index(
                files,
                recursive=recursive,
                force=force,
                on_file_indexed=_on_file_indexed,
            )
    except Exception as exc:
        _fail("Indexing failed", exc)
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]Indexing complete.[/bold green] Indexed {count} chunk(s).")
    if count == 0:
        console.print(
            "[dim]Nothing new to index — matching files were already stored. "
            "Use --force to re-index them.[/dim]"
        )
    console.print(f"[dim]Store: {store_dir}[/dim]")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Text query to search for.")],
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of top results to return."),
    ] = 5,
    store_dir: StoreDirOption = Path(".echovector"),
    backend: BackendOption = "clap",
    device: DeviceOption = None,
    min_score: Annotated[
        float | None,
        typer.Option("--min-score", help="Discard results scoring below this threshold."),
    ] = None,
    merge_spans: Annotated[
        bool,
        typer.Option(
            "--merge-spans/--no-merge-spans",
            help="Fold consecutive hits from one file into a single moment.",
        ),
    ] = True,
    max_gap_seconds: Annotated[
        float,
        typer.Option("--max-gap", help="Largest silence still counted as one moment."),
    ] = 0.0,
    max_per_file: Annotated[
        int | None,
        typer.Option("--max-per-file", help="Cap how many results any one file contributes."),
    ] = None,
) -> None:
    """Search an existing vector index without scanning audio files."""
    if not _require_index(store_dir):
        raise typer.Exit(code=1)

    try:
        engine = EchoVector(
            store_dir=store_dir, backend=backend, **_backend_kwargs(backend, device)
        )
        filters = SearchFilter(min_score=min_score) if min_score is not None else None
        results = engine.search(
            query,
            top_k=top_k,
            filters=filters,
            merge_spans=merge_spans,
            max_gap_seconds=max_gap_seconds,
            max_per_file=max_per_file,
        )
    except Exception as exc:
        _fail("Search failed", exc)
        raise typer.Exit(code=1) from exc

    _print_results(results, title=f"Top {len(results)} Results for '{query}'")


@app.command()
def similar(
    audio_path: Annotated[Path, typer.Argument(help="Audio file to find matches for.")],
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of top results to return."),
    ] = 5,
    store_dir: StoreDirOption = Path(".echovector"),
    backend: BackendOption = "clap",
    device: DeviceOption = None,
    include_source: Annotated[
        bool,
        typer.Option("--include-source", help="Keep hits from the query file itself."),
    ] = False,
) -> None:
    """Find indexed audio that sounds like a given clip.

    Uses no text at all, so it also works with audio-only backends such as
    wav2vec2, hubert, and ast.
    """
    if not _require_index(store_dir):
        raise typer.Exit(code=1)

    try:
        engine = EchoVector(
            store_dir=store_dir, backend=backend, **_backend_kwargs(backend, device)
        )
        results = engine.search_by_audio(
            audio_path, top_k=top_k, exclude_source=not include_source
        )
    except Exception as exc:
        _fail("Similarity search failed", exc)
        raise typer.Exit(code=1) from exc

    _print_results(results, title=f"Sounds like '{audio_path.name}'")


@app.command()
def extract(
    audio_path: Annotated[Path, typer.Argument(help="Source audio file.")],
    start: Annotated[float, typer.Argument(help="Start time in seconds.")],
    end: Annotated[float, typer.Argument(help="End time in seconds.")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="Where to write the clip.")],
    padding: Annotated[
        float,
        typer.Option("--padding", help="Extra seconds of context on each side."),
    ] = 0.0,
) -> None:
    """Write the audio between two timestamps to a new file."""
    try:
        written = extract_clip(audio_path, start, end, out_path=out_path, padding=padding)
    except Exception as exc:
        _fail("Extract failed", exc)
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]Wrote[/bold green] {written}")


@app.command()
def play(
    audio_path: Annotated[Path, typer.Argument(help="Source audio file.")],
    start: Annotated[float, typer.Argument(help="Start time in seconds.")],
    end: Annotated[float, typer.Argument(help="End time in seconds.")],
    player: Annotated[
        str,
        typer.Option("--player", help=f"One of: {', '.join(SUPPORTED_PLAYERS)}."),
    ] = "ffplay",
) -> None:
    """Print the command that plays a search hit.

    The command is printed rather than run, so nothing is executed on your
    behalf and the line can be piped, copied, or edited first.
    """
    try:
        argv = player_command(audio_path, start, end, player=player)
    except Exception as exc:
        _fail("Cannot build a play command", exc)
        raise typer.Exit(code=1) from exc

    console.print(shlex.join(argv))


@app.command(name="eval")
def evaluate_index(
    golden_set: Annotated[Path, typer.Argument(help="Golden-set JSON file.")],
    store_dir: StoreDirOption = Path(".echovector"),
    backend: BackendOption = "clap",
    device: DeviceOption = None,
    ks: Annotated[
        str,
        typer.Option("--ks", help="Comma-separated cut-offs to score at."),
    ] = "1,5,10",
    min_overlap: Annotated[
        float,
        typer.Option("--min-overlap", help="Overlap fraction required to count as a hit."),
    ] = 0.0,
    merge_spans: Annotated[
        bool,
        typer.Option("--merge-spans/--no-merge-spans", help="Score merged moments."),
    ] = True,
    per_query: Annotated[
        bool,
        typer.Option("--per-query", help="Show a row per query as well as the summary."),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the report as JSON."),
    ] = False,
) -> None:
    """Measure retrieval quality against a golden set.

    Without this, a change to chunk size, backend, or index type is a guess:
    nothing else in the tool says whether search got better or worse.
    """
    if not _require_index(store_dir):
        raise typer.Exit(code=1)

    try:
        cutoffs = tuple(int(part) for part in ks.split(",") if part.strip())
        if not cutoffs:
            raise ValueError("--ks needs at least one cut-off, e.g. --ks 1,5,10")

        engine = EchoVector(
            store_dir=store_dir, backend=backend, **_backend_kwargs(backend, device)
        )
        report = evaluate(
            lambda query, k: engine.search(query, top_k=k, merge_spans=merge_spans),
            load_golden_set(golden_set),
            ks=cutoffs,
            judge=RelevanceJudge(min_overlap_fraction=min_overlap),
        )
    except Exception as exc:
        _fail("Evaluation failed", exc)
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(json.dumps(report.to_dict()))
    else:
        console.print(report.format_table(include_per_query=per_query))


def _print_results(results: list[SearchResult], title: str) -> None:
    """Render search results, or say plainly that there were none."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=title)
    table.add_column("Score", justify="right", style="cyan", no_wrap=True)
    table.add_column("File", style="magenta")
    table.add_column("Timestamp", justify="right", style="green")
    table.add_column("Chunks", justify="right", style="dim")

    for result in results:
        table.add_row(
            f"{result.score:.4f}",
            str(result.metadata.get("filename", result.filepath)),
            f"{result.timestamp_range.start:.1f}s-{result.timestamp_range.end:.1f}s",
            str(result.metadata.get("merged_chunk_count", 1)),
        )

    console.print(table)


@app.command()
def watch(
    files: Annotated[
        list[Path],
        typer.Argument(help="Audio files or directories to keep indexed."),
    ],
    recursive: Annotated[
        bool,
        typer.Option("--recursive/--no-recursive", "-r", help="Search directories recursively."),
    ] = True,
    store_dir: StoreDirOption = Path(".echovector"),
    backend: BackendOption = "clap",
    device: DeviceOption = None,
    interval: Annotated[
        float,
        typer.Option("--interval", help="Seconds between rescans."),
    ] = DEFAULT_INTERVAL_SECONDS,
    chunk_seconds: Annotated[
        float,
        typer.Option("--chunk-seconds", help="Audio chunk size to embed."),
    ] = DEFAULT_CHUNK_SECONDS,
) -> None:
    """Keep an index current as a folder changes, until interrupted.

    Files that appear are indexed, files that change are re-read, and files that
    are deleted are dropped. A file still being copied in is left alone until it
    stops changing, so a half-written recording is never indexed as if complete.
    """
    symbols = {"added": ("+", "green"), "changed": ("~", "yellow"), "removed": ("-", "red")}

    def _report(event: WatchEvent) -> None:
        if event.kind == "failed":
            detail = escape(event.error or "")
            console.print(f"[bold red]![/bold red] {event.filepath.name} [dim]{detail}[/dim]")
            return
        mark, colour = symbols[event.kind]
        detail = f" [dim]({event.chunks} chunks)[/dim]" if event.chunks else ""
        console.print(f"[{colour}]{mark}[/{colour}] {event.filepath.name}{detail}")

    try:
        engine = EchoVector(
            store_dir=store_dir,
            backend=backend,
            chunk_seconds=chunk_seconds,
            **_backend_kwargs(backend, device),
        )
        watcher = FolderWatcher(engine, files, recursive=recursive)
        watched = len(engine.resolve_audio_files(files, recursive=recursive))
        console.print(
            f"[bold green]Watching[/bold green] {watched} file(s), rescanning every "
            f"{interval:g}s. [dim]Ctrl-C to stop.[/dim]"
        )
        watcher.run(interval=interval, on_event=_report)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
    except Exception as exc:
        _fail("Watch failed", exc)
        raise typer.Exit(code=1) from exc


@app.command()
def prune(
    store_dir: StoreDirOption = Path(".echovector"),
    backend: BackendOption = "clap",
    device: DeviceOption = None,
    check_only: Annotated[
        bool,
        typer.Option("--check", help="Report what would change without touching the index."),
    ] = False,
) -> None:
    """Drop deleted files from the index and report ones that changed.

    Files are matched by content signature, so a recording edited since it was
    indexed is reported here and re-read on the next `index` run.
    """
    if not _require_index(store_dir):
        raise typer.Exit(code=1)

    try:
        engine = EchoVector(
            store_dir=store_dir, backend=backend, **_backend_kwargs(backend, device)
        )
        stale = engine.stale_files()
        removed = [] if check_only else engine.prune()
        if check_only:
            removed = [
                path
                for path in engine.index_backend.store.list_filepaths()
                if not Path(path).exists()
            ]
    except Exception as exc:
        _fail("Prune failed", exc)
        raise typer.Exit(code=1) from exc

    verb = "Would remove" if check_only else "Removed"
    console.print(f"[bold green]{verb}[/bold green] {len(removed)} deleted file(s).")
    for path in removed:
        console.print(f"  [dim]-[/dim] {path}")

    if stale:
        console.print(f"[yellow]{len(stale)} file(s) changed since indexing:[/yellow]")
        for path in stale:
            console.print(f"  [dim]~[/dim] {path}")
        console.print("[dim]Re-run 'echovector index' on them to refresh.[/dim]")


@app.command()
def stats(
    store_dir: StoreDirOption = Path(".echovector"),
    backend: BackendOption = "clap",
    device: DeviceOption = None,
) -> None:
    """Display statistics about the current index."""
    if not _require_index(store_dir):
        raise typer.Exit(code=1)

    try:
        engine = EchoVector(
            store_dir=store_dir, backend=backend, **_backend_kwargs(backend, device)
        )
        stats_data = engine.stats()
    except Exception as exc:
        _fail("Stats failed", exc)
        raise typer.Exit(code=1) from exc

    table = Table(title="Index Statistics", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in stats_data.items():
        table.add_row(key, str(value))

    console.print(table)


@app.command()
def version() -> None:
    """Print the installed EchoVector version."""
    console.print(__version__)


def _require_index(store_dir: Path) -> bool:
    """Report a missing index clearly instead of returning an empty result set.

    An unindexed directory and a query with no matches used to look identical.
    """
    if IndexManifest.path_for(store_dir).exists() or (store_dir / "index.faiss").exists():
        return True
    console.print(
        f"[bold red]No index found at[/bold red] {store_dir}\n"
        f"[dim]Run 'echovector index <audio-path> --store-dir {store_dir}' first.[/dim]"
    )
    return False


if __name__ == "__main__":
    app()
