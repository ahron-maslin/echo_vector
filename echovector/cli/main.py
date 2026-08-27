"""Command-line interface for EchoVector."""

import inspect
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from echovector import EchoVector, SearchFilter, __version__
from echovector.embeddings.factory import backend_class_for
from echovector.indexing.manifest import IndexManifest

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
    ] = 10.0,
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
        console.print(f"[bold red]Indexing failed:[/bold red] {exc}")
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
) -> None:
    """Search an existing vector index without scanning audio files."""
    if not _require_index(store_dir):
        raise typer.Exit(code=1)

    try:
        engine = EchoVector(
            store_dir=store_dir, backend=backend, **_backend_kwargs(backend, device)
        )
        filters = SearchFilter(min_score=min_score) if min_score is not None else None
        results = engine.search(query, top_k=top_k, filters=filters)
    except Exception as exc:
        console.print(f"[bold red]Search failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Top {len(results)} Results for '{query}'")
    table.add_column("Score", justify="right", style="cyan", no_wrap=True)
    table.add_column("File", style="magenta")
    table.add_column("Timestamp", justify="right", style="green")

    for result in results:
        table.add_row(
            f"{result.score:.4f}",
            str(result.metadata.get("filename", result.filepath)),
            f"{result.timestamp_range.start:.1f}s-{result.timestamp_range.end:.1f}s",
        )

    console.print(table)


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
        console.print(f"[bold red]Stats failed:[/bold red] {exc}")
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
