"""FastAPI server for EchoVector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from echovector.core import EchoVector

app = FastAPI(
    title="EchoVector API",
    description="Semantic search over audio files.",
    version="0.1.0",
)

# Default engine instance — override via app.dependency_overrides in tests.
_default_engine: EchoVector | None = None


def get_engine() -> EchoVector:
    """Return the active EchoVector engine."""
    if _default_engine is None:
        raise RuntimeError(
            "No EchoVector engine configured. Call configure_engine() before starting the server."
        )
    return _default_engine


def configure_engine(engine: EchoVector) -> None:
    """Set the engine used by the server at startup."""
    global _default_engine
    _default_engine = engine


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    """Paths to index."""

    paths: list[str] = Field(..., description="Audio file or directory paths to index.")
    force: bool = Field(False, description="Re-index files that are already stored.")


class IndexResponse(BaseModel):
    """Result of an index operation."""

    chunks_added: int
    files_skipped: int


class SearchRequest(BaseModel):
    """Text query for audio search."""

    query: str = Field(..., description="Natural language query.")
    top_k: int = Field(5, ge=1, description="Maximum results to return.")


class SearchResultItem(BaseModel):
    """Single search result."""

    filepath: str
    start: float
    end: float
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search results."""

    results: list[SearchResultItem]


class StatsResponse(BaseModel):
    """Index statistics."""

    chunks: int
    embedding_dim: int
    store_dir: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/index", response_model=IndexResponse)
def index_audio(
    request: IndexRequest,
    engine: EchoVector = Depends(get_engine),
) -> IndexResponse:
    """Index audio files or directories."""
    resolved_files = engine.resolve_audio_files(request.paths)
    files_skipped = 0 if request.force else sum(1 for f in resolved_files if engine.is_indexed(f))
    chunks_added = engine.index(request.paths, force=request.force)
    return IndexResponse(chunks_added=chunks_added, files_skipped=files_skipped)


@app.post("/search", response_model=SearchResponse)
def search_audio(
    request: SearchRequest,
    engine: EchoVector = Depends(get_engine),
) -> SearchResponse:
    """Search indexed audio with a text query."""
    results = engine.search(request.query, top_k=request.top_k)
    return SearchResponse(
        results=[
            SearchResultItem(
                filepath=r.filepath,
                start=r.timestamp_range.start,
                end=r.timestamp_range.end,
                score=r.score,
                metadata=r.metadata or {},
            )
            for r in results
        ]
    )


@app.get("/stats", response_model=StatsResponse)
def get_stats(engine: EchoVector = Depends(get_engine)) -> StatsResponse:
    """Return index statistics."""
    s = engine.stats()
    return StatsResponse(
        chunks=int(s["chunks"]),
        embedding_dim=int(s["embedding_dim"]),
        store_dir=str(s["store_dir"]),
    )


@app.post("/reset")
def reset_index(engine: EchoVector = Depends(get_engine)) -> dict[str, str]:
    """Clear the index."""
    engine.reset()
    return {"status": "ok"}
