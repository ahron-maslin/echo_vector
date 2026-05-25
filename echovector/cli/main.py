"""
Command-line interface for EchoVector using Typer and Rich.
"""
import time
from typing import List

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

app = typer.Typer(
    name="echovector",
    help="EchoVector CLI for semantic text search over audio files.",
    add_completion=False,
)
console = Console()

@app.command()
def index(
    files: List[str] = typer.Argument(..., help="Audio files or directories to index."),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Recursively search directories for audio files."
    ),
) -> None:
    """
    Index audio files for semantic search.
    """
    console.print(f"[bold green]Starting indexing for {len(files)} target(s)...[/bold green]")
    
    # Simulate work with a progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task1 = progress.add_task("[cyan]Processing files...", total=100)
        
        for _ in range(100):
            time.sleep(0.01)  # Mock work
            progress.update(task1, advance=1)
            
    console.print("[bold green]✓ Indexing complete.[/bold green]")

@app.command()
def search(
    query: str = typer.Argument(..., help="Text query to search for."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top results to return."),
) -> None:
    """
    Search for a text query within indexed audio files.
    """
    console.print(f"[bold blue]Searching for:[/bold blue] '{query}'")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Searching index...", total=None)
        time.sleep(0.5)  # Mock search time
        progress.update(task, completed=100)
        
    table = Table(title=f"Top {top_k} Results for '{query}'")
    table.add_column("Score", justify="right", style="cyan", no_wrap=True)
    table.add_column("File", style="magenta")
    table.add_column("Timestamp", justify="right", style="green")
    
    # Simulate finding results
    for i in range(min(top_k, 3)):
        score = 0.95 - (i * 0.05)
        table.add_row(f"{score:.4f}", f"audio_file_{i+1}.wav", f"{(i+1)*10.5:.1f}s")
        
    if top_k == 0:
        console.print("[yellow]No results requested.[/yellow]")
    else:
        console.print(table)

@app.command()
def stats() -> None:
    """
    Display statistics about the current index.
    """
    console.print("[bold yellow]Index Statistics[/bold yellow]")
    
    table = Table(show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    # Mock stats
    table.add_row("Total Files Indexed", "42")
    table.add_row("Total Audio Duration", "14h 32m")
    table.add_row("Index Size", "156 MB")
    
    console.print(table)

if __name__ == "__main__":
    app()
