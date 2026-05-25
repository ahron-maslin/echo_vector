"""
End-to-end tests for the EchoVector CLI.
"""
from typer.testing import CliRunner

from echovector.cli.main import app

runner = CliRunner()

def test_index_command() -> None:
    """Test the index command."""
    result = runner.invoke(app, ["index", "dummy_file.wav"])
    assert result.exit_code == 0
    assert "Starting indexing for 1 target(s)..." in result.stdout
    assert "Indexing complete." in result.stdout

def test_search_command() -> None:
    """Test the search command."""
    result = runner.invoke(app, ["search", "hello world", "--top-k", "2"])
    assert result.exit_code == 0
    assert "Searching for:" in result.stdout
    assert "hello world" in result.stdout
    assert "Top 2 Results" in result.stdout

def test_stats_command() -> None:
    """Test the stats command."""
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Index Statistics" in result.stdout
    assert "Total Files Indexed" in result.stdout
    assert "Total Audio Duration" in result.stdout
    assert "Index Size" in result.stdout

def test_help_command() -> None:
    """Test the help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "EchoVector CLI" in result.stdout
    assert "index" in result.stdout
    assert "search" in result.stdout
    assert "stats" in result.stdout
