"""Command-line interface for EchoVector."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from echovector import EchoVector

app = typer.Typer(
    name="echovector",
    help="EchoVector CLI for semantic text search over audio files.",
    add_completion=False,
)
console = Console()


def _open_engine(store_dir: Path, backend: str, device: str | None) -> EchoVector:
    if device:
        return EchoVector(store_dir=store_dir, backend=backend, device=device)
    return EchoVector(store_dir=store_dir, backend=backend)


@app.command()
def index(
    files: Annotated[
        list[Path],
        typer.Argument(help="Audio files or directories to index."),
    ],
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Recursively search directories for audio files."),
    ] = False,
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory for index files."),
    ] = Path(".echovector"),
    backend: Annotated[str, typer.Option("--backend", help="Embedding backend to use.")] = "clap",
    device: Annotated[
        str | None,
        typer.Option("--device", help="Model device, e.g. cpu or cuda."),
    ] = None,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Clear the existing index before indexing."),
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
        if device:
            engine = EchoVector(
                store_dir=store_dir,
                backend=backend,
                device=device,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
            )
        else:
            engine = EchoVector(
                store_dir=store_dir,
                backend=backend,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
            )
        if reset:
            engine.reset()
        count = engine.index(files, recursive=recursive)
    except Exception as exc:
        console.print(f"[bold red]Indexing failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]Indexing complete.[/bold green] Indexed {count} chunk(s).")
    console.print(f"[dim]Store: {store_dir}[/dim]")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Text query to search for.")],
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of top results to return."),
    ] = 5,
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory containing the index."),
    ] = Path(".echovector"),
    backend: Annotated[
        str,
        typer.Option("--backend", help="Embedding backend used for the index."),
    ] = "clap",
    device: Annotated[
        str | None,
        typer.Option("--device", help="Model device, e.g. cpu or cuda."),
    ] = None,
) -> None:
    """Search an existing vector index without scanning audio files."""
    try:
        engine = _open_engine(store_dir=store_dir, backend=backend, device=device)
        results = engine.search(query, top_k=top_k)
    except Exception as exc:
        console.print(f"[bold red]Search failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"Top {top_k} Results for '{query}'")
    table.add_column("Score", justify="right", style="cyan", no_wrap=True)
    table.add_column("File", style="magenta")
    table.add_column("Timestamp", justify="right", style="green")

    for result in results:
        table.add_row(
            f"{result.score:.4f}",
            str(result.metadata.get("filename", result.filepath)),
            f"{result.timestamp_range.start:.1f}s-{result.timestamp_range.end:.1f}s",
        )

    if results:
        console.print(table)
    else:
        console.print("[yellow]No results found.[/yellow]")


@app.command()
def stats(
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory containing the index."),
    ] = Path(".echovector"),
    backend: Annotated[
        str,
        typer.Option("--backend", help="Embedding backend used for the index."),
    ] = "clap",
    device: Annotated[
        str | None,
        typer.Option("--device", help="Model device, e.g. cpu or cuda."),
    ] = None,
) -> None:
    """Display statistics about the current index."""
    try:
        stats_data = _open_engine(store_dir=store_dir, backend=backend, device=device).stats()
    except Exception as exc:
        console.print(f"[bold red]Stats failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Index Statistics", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in stats_data.items():
        table.add_row(key, str(value))

    console.print(table)


if __name__ == "__main__":
    app()
