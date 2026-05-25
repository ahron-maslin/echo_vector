"""
API server module for EchoVector using FastAPI.
"""
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="EchoVector API",
    description="API for indexing and searching audio files.",
    version="0.1.0",
)

class IndexRequest(BaseModel):
    """Request model for indexing an audio file."""
    file_path: str = Field(..., description="Path to the audio file to index")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class IndexResponse(BaseModel):
    """Response model for indexing operations."""
    status: str
    file_path: str
    message: str

class SearchRequest(BaseModel):
    """Request model for searching audio files."""
    query: str = Field(..., description="Text query to search for in audio files")
    top_k: int = Field(10, description="Number of results to return")

class SearchResult(BaseModel):
    """Individual search result model."""
    file_path: str
    score: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    """Response model for search operations."""
    results: List[SearchResult]
    total: int
    query_time_ms: float

class StatsResponse(BaseModel):
    """Response model for index statistics."""
    total_files: int
    total_duration_seconds: float
    index_size_bytes: int

# In-memory mock database for the server
_db: Dict[str, Any] = {}

@app.post("/index", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def index_audio(request: IndexRequest) -> IndexResponse:
    """
    Index a new audio file for semantic search.
    """
    _db[request.file_path] = {
        "metadata": request.metadata,
        "duration": 120.0  # Mock duration
    }
    return IndexResponse(
        status="success",
        file_path=request.file_path,
        message="Audio file indexed successfully"
    )

@app.post("/search", response_model=SearchResponse)
async def search_audio(request: SearchRequest) -> SearchResponse:
    """
    Search for text query within indexed audio files.
    """
    results = []
    # Mock search logic returning predefined results based on the index contents
    for file_path, data in _db.items():
        results.append(
            SearchResult(
                file_path=file_path,
                score=0.95,
                timestamp=12.5,
                metadata=data.get("metadata", {})
            )
        )
    
    # Sort and truncate
    results.sort(key=lambda x: x.score, reverse=True)
    results = results[:request.top_k]
    
    return SearchResponse(
        results=results,
        total=len(results),
        query_time_ms=45.6
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    Get index statistics.
    """
    total_files = len(_db)
    total_duration = sum(d.get("duration", 0.0) for d in _db.values())
    return StatsResponse(
        total_files=total_files,
        total_duration_seconds=total_duration,
        index_size_bytes=total_files * 1024  # Mock size
    )
