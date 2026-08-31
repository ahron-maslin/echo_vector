"""FastAPI server for EchoVector.

Security posture
----------------
Authentication is *optional* and off by default. Set ``ECHOVECTOR_API_TOKEN`` to
require ``Authorization: Bearer <token>`` on the routes that mutate the index or
disclose the paths handed to it (``POST /index``, ``POST /reset``, ``GET /jobs``,
``GET /jobs/{job_id}``); leave it unset and those routes stay open, which is the
historical local-use behaviour. ``GET /health``, ``POST /search`` and
``GET /stats`` are never token-gated.

A token limits *who* can call the mutating routes. It does not make them safe to
expose publicly: ``POST /index`` still reads whatever filesystem paths the request
body names, so anyone holding the token can make the server read any file the
server process can read, and ``POST /reset`` still deletes the index irreversibly.
``GET /stats`` discloses the store directory to unauthenticated callers. Bind to
localhost, or put the server behind an authenticating proxy on a host whose
filesystem you are happy to expose to the token holder.

Concurrency
-----------
``EchoVector`` owns a SQLite connection and a FAISS index and mutates both in
place, so it is not safe for concurrent writers, nor for a reader racing a
writer. All engine access therefore goes through a single lock held by
:class:`IndexJobManager`. Indexing runs on a background thread one file at a
time, taking that lock per file and releasing it in between, so ``POST /search``
and ``GET /stats`` keep working during a long job -- they wait at most for the
current file, and observe a consistent view of the files indexed so far. Only one
index job may be in flight at a time; a second request is rejected with 409.
"""

from __future__ import annotations

import copy
import os
import secrets
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from echovector import __version__

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from concurrent.futures import Future
    from pathlib import Path

    from echovector.core import EchoVector

TOKEN_ENV_VAR = "ECHOVECTOR_API_TOKEN"  # noqa: S105 - the variable name, not a secret

# Jobs are held in memory; cap the history so a long-lived server does not grow
# without bound.
MAX_JOB_HISTORY = 100

app = FastAPI(
    title="EchoVector API",
    description="Semantic search over audio files.",
    version=__version__,
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
# Authentication
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(
    auto_error=False,
    description=f"Bearer token; configured via the {TOKEN_ENV_VAR} environment variable.",
)


def require_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """Enforce bearer-token auth when ``ECHOVECTOR_API_TOKEN`` is set.

    The environment variable is read per request so that a deployment can rotate
    the token without restarting, and so tests can set and clear it.

    Args:
        credentials: Bearer credentials parsed from the ``Authorization`` header,
            or None when the header is absent or not a bearer challenge.

    Raises:
        HTTPException: 401 if a token is configured and the request does not
            present a matching one.
    """
    expected = os.environ.get(TOKEN_ENV_VAR)
    if not expected:
        return

    presented = credentials.credentials if credentials is not None else ""
    # compare_digest over bytes: the str overload rejects non-ASCII tokens.
    if not secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def auth_required() -> bool:
    """Return True if a token is currently configured for the mutating routes."""
    return bool(os.environ.get(TOKEN_ENV_VAR))


# ---------------------------------------------------------------------------
# Background index jobs
# ---------------------------------------------------------------------------


class JobStatus(StrEnum):
    """Lifecycle state of an index job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})


class IndexJobConflictError(RuntimeError):
    """Raised when an index job is submitted while another one is in flight."""

    def __init__(self, active_job_id: str) -> None:
        """Record the job that is currently holding the engine.

        Args:
            active_job_id: Identifier of the in-flight job.
        """
        self.active_job_id = active_job_id
        super().__init__(f"Index job {active_job_id} is already running.")


@dataclass
class IndexJob:
    """Mutable record of a single index job.

    Instances live only in the owning :class:`IndexJobManager`; callers receive
    deep copies so they never observe a half-written update.
    """

    job_id: str
    paths: list[str]
    force: bool
    created_at: datetime
    status: JobStatus = JobStatus.QUEUED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    files_total: int = 0
    files_done: int = 0
    chunks_added: int = 0
    files_skipped: int = 0
    current_file: str | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Return True once the job has finished, successfully or not."""
        return self.status in TERMINAL_STATUSES


def _now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


class IndexJobManager:
    """Run index jobs one at a time on a background thread.

    Jobs are stored **in memory only**: restarting the server loses every job
    record, including the history of completed and failed jobs. Only the last
    :data:`MAX_JOB_HISTORY` jobs are retained.

    The manager also owns the single lock through which all engine access must
    pass, exposed as :meth:`engine_access`. The job worker takes it once per file
    rather than once per job, so read routes are never blocked for longer than
    the file currently being indexed.
    """

    def __init__(self, executor: ThreadPoolExecutor | None = None) -> None:
        """Create a manager.

        Args:
            executor: Executor used to run jobs. Must be single-threaded; the
                default is a dedicated one-worker pool. Injectable for tests.
        """
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="echovector-index"
        )
        self._lock = threading.Lock()
        self._engine_lock = threading.RLock()
        self._jobs: dict[str, IndexJob] = {}
        self._order: list[str] = []
        self._futures: dict[str, Future[None]] = {}
        self._active_job_id: str | None = None

    @contextmanager
    def engine_access(self) -> Iterator[None]:
        """Hold the lock that serialises every access to the engine.

        Read routes use this so they never race the background writer. A read
        waits at most for the file the job is currently on, not for the whole
        job, because the worker re-acquires the lock per file.

        Yields:
            None, with the engine lock held.
        """
        with self._engine_lock:
            yield

    @property
    def active_job_id(self) -> str | None:
        """Return the id of the in-flight job, or None when the engine is idle."""
        with self._lock:
            return self._active_job_id

    def submit(
        self,
        engine: EchoVector,
        paths: Sequence[str],
        force: bool,
    ) -> IndexJob:
        """Queue an index job and return its initial record.

        Args:
            engine: Engine to index into.
            paths: Filesystem paths (files or directories) to index.
            force: Re-index files that are already stored.

        Returns:
            A snapshot of the newly created job.

        Raises:
            IndexJobConflictError: If another job is already in flight.
        """
        with self._lock:
            if self._active_job_id is not None:
                raise IndexJobConflictError(self._active_job_id)

            job = IndexJob(
                job_id=str(uuid.uuid4()),
                paths=list(paths),
                force=force,
                created_at=_now(),
            )
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            self._active_job_id = job.job_id
            self._trim_history_locked()
            self._futures[job.job_id] = self._executor.submit(
                self._run_job, job.job_id, engine, list(paths), force
            )
            return copy.deepcopy(job)

    def get(self, job_id: str) -> IndexJob | None:
        """Return a snapshot of one job, or None if it is unknown or evicted.

        Args:
            job_id: Identifier returned by :meth:`submit`.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job is not None else None

    def list_jobs(self) -> list[IndexJob]:
        """Return snapshots of the retained jobs, most recently submitted first."""
        with self._lock:
            return [copy.deepcopy(self._jobs[job_id]) for job_id in reversed(self._order)]

    def wait(self, job_id: str, timeout: float | None = None) -> IndexJob | None:
        """Block until a job reaches a terminal state and return its snapshot.

        Args:
            job_id: Identifier returned by :meth:`submit`.
            timeout: Seconds to wait, or None to wait indefinitely.

        Returns:
            A snapshot of the job, or None if it is unknown.

        Raises:
            TimeoutError: If the job is still running when the timeout expires.
        """
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout)
        return self.get(job_id)

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting jobs and release the worker thread.

        Args:
            wait: Block until the in-flight job finishes.
        """
        self._executor.shutdown(wait=wait)

    # -- internals ---------------------------------------------------------

    def _trim_history_locked(self) -> None:
        """Evict the oldest terminal jobs once history exceeds the cap."""
        while len(self._order) > MAX_JOB_HISTORY:
            evictable = next(
                (job_id for job_id in self._order if self._jobs[job_id].is_terminal),
                None,
            )
            if evictable is None:
                return
            self._order.remove(evictable)
            del self._jobs[evictable]
            self._futures.pop(evictable, None)

    def _run_job(
        self,
        job_id: str,
        engine: EchoVector,
        paths: list[str],
        force: bool,
    ) -> None:
        """Execute one index job, recording progress and any failure on the record."""
        job = self._jobs[job_id]
        with self._lock:
            job.status = JobStatus.RUNNING
            job.started_at = _now()
        try:
            self._index_files(job, engine, paths, force)
        # Any failure belongs on the job record: the client is no longer holding
        # the request that started this work.
        except Exception as exc:
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
        else:
            with self._lock:
                job.status = JobStatus.COMPLETED
        finally:
            with self._lock:
                job.current_file = None
                job.finished_at = _now()
                self._active_job_id = None

    def _index_files(
        self,
        job: IndexJob,
        engine: EchoVector,
        paths: list[str],
        force: bool,
    ) -> None:
        """Index each resolved file in turn, releasing the engine lock in between."""
        with self._engine_lock:
            files = engine.resolve_audio_files(paths)
            skipped = 0 if force else sum(1 for f in files if engine.is_indexed(f))

        with self._lock:
            job.files_total = len(files)
            job.files_skipped = skipped

        for position, file_path in enumerate(files):
            with self._lock:
                job.current_file = str(file_path)

            def on_file_indexed(
                files_done: int,
                _files_total: int,
                current: Path,
                *,
                offset: int = position,
            ) -> None:
                with self._lock:
                    job.files_done = offset + files_done
                    job.current_file = str(current)

            with self._engine_lock:
                added = engine.index([file_path], force=force, on_file_indexed=on_file_indexed)

            with self._lock:
                job.chunks_added += added
                # The engine skips already-indexed files without invoking the
                # callback, so advance the counter here too.
                job.files_done = max(job.files_done, position + 1)


_default_job_manager = IndexJobManager()


def get_job_manager() -> IndexJobManager:
    """Return the job manager backing the index routes."""
    return _default_job_manager


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    """Paths to index."""

    paths: list[str] = Field(..., description="Audio file or directory paths to index.")
    force: bool = Field(False, description="Re-index files that are already stored.")


class IndexJobResponse(BaseModel):
    """State of an index job.

    Job records live in the server process only and are lost on restart.
    """

    job_id: str
    status: JobStatus
    paths: list[str]
    force: bool
    files_total: int
    files_done: int
    files_skipped: int
    chunks_added: int
    progress: float = Field(..., ge=0.0, le=1.0, description="Fraction of files processed.")
    current_file: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def from_job(cls, job: IndexJob) -> Self:
        """Build a response from a job snapshot.

        Args:
            job: Snapshot returned by the job manager.
        """
        if job.is_terminal:
            progress = 1.0
        elif job.files_total:
            progress = min(job.files_done / job.files_total, 1.0)
        else:
            progress = 0.0
        return cls(
            job_id=job.job_id,
            status=job.status,
            paths=job.paths,
            force=job.force,
            files_total=job.files_total,
            files_done=job.files_done,
            files_skipped=job.files_skipped,
            chunks_added=job.chunks_added,
            progress=progress,
            current_file=job.current_file,
            error=job.error,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )


class JobListResponse(BaseModel):
    """Known index jobs, most recently submitted first."""

    jobs: list[IndexJobResponse]


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


class ResetResponse(BaseModel):
    """Outcome of clearing the index."""

    status: str = Field(..., description="Always 'ok' when the index was cleared.")
    chunks_removed: int = Field(..., description="Chunks that were in the index before the reset.")


class HealthResponse(BaseModel):
    """Liveness of the server itself, independent of the engine."""

    status: str
    version: str
    engine_configured: bool
    auth_required: bool
    active_job_id: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health(
    manager: IndexJobManager = Depends(get_job_manager),
) -> HealthResponse:
    """Report server liveness without requiring a configured engine or a token."""
    try:
        get_engine()
    except RuntimeError:
        engine_configured = False
    else:
        engine_configured = True

    return HealthResponse(
        status="ok",
        version=__version__,
        engine_configured=engine_configured,
        auth_required=auth_required(),
        active_job_id=manager.active_job_id,
    )


@app.post(
    "/index",
    response_model=IndexJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_token)],
)
def index_audio(
    request: IndexRequest,
    response: Response,
    engine: EchoVector = Depends(get_engine),
    manager: IndexJobManager = Depends(get_job_manager),
    wait: bool = Query(
        False,
        description="Run the job inline and return only once it finishes. Small jobs only.",
    ),
) -> IndexJobResponse:
    """Start indexing audio files or directories.

    Returns 202 with a job id immediately; poll ``GET /jobs/{job_id}`` for
    progress. Pass ``?wait=true`` to block until the job finishes and get 200
    with the final counts instead -- convenient for a handful of files, but a
    real corpus will outlast any client timeout.

    Args:
        request: Paths to index and whether to force re-indexing.
        response: Injected so the status code can drop to 200 for ``wait=true``.
        engine: Engine to index into.
        manager: Job manager owning the worker thread and the engine lock.
        wait: Run the job inline instead of in the background.

    Returns:
        The job record, freshly queued or already finished.

    Raises:
        HTTPException: 409 if another index job is in flight, or 500 if an
            inline (``wait=true``) job failed.
    """
    try:
        job = manager.submit(engine, request.paths, request.force)
    except IndexJobConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Index job {exc.active_job_id} is already running. EchoVector allows only one "
                f"writer at a time, so this request was rejected rather than queued: wait for "
                f"GET /jobs/{exc.active_job_id} to report 'completed' or 'failed', then retry."
            ),
        ) from exc

    if not wait:
        return IndexJobResponse.from_job(job)

    finished = manager.wait(job.job_id) or job
    if finished.status is JobStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=finished.error or "Index job failed.",
        )
    response.status_code = status.HTTP_200_OK
    return IndexJobResponse.from_job(finished)


@app.get("/jobs", response_model=JobListResponse, dependencies=[Depends(require_token)])
def list_jobs(
    manager: IndexJobManager = Depends(get_job_manager),
) -> JobListResponse:
    """List retained index jobs, most recent first.

    Args:
        manager: Job manager holding the in-memory job history.
    """
    return JobListResponse(jobs=[IndexJobResponse.from_job(job) for job in manager.list_jobs()])


@app.get("/jobs/{job_id}", response_model=IndexJobResponse, dependencies=[Depends(require_token)])
def get_job(
    job_id: str,
    manager: IndexJobManager = Depends(get_job_manager),
) -> IndexJobResponse:
    """Return the current state of one index job.

    Args:
        job_id: Identifier returned by ``POST /index``.
        manager: Job manager holding the in-memory job history.

    Raises:
        HTTPException: 404 if the job is unknown, or was evicted from history.
    """
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown job {job_id}. Jobs are held in memory and do not survive a restart.",
        )
    return IndexJobResponse.from_job(job)


@app.post("/search", response_model=SearchResponse)
def search_audio(
    request: SearchRequest,
    engine: EchoVector = Depends(get_engine),
    manager: IndexJobManager = Depends(get_job_manager),
) -> SearchResponse:
    """Search indexed audio with a text query.

    Safe to call while an index job runs: the engine lock is taken for the query,
    so the search sees the files indexed so far and never races the writer.

    Args:
        request: Query text and result count.
        engine: Engine to search.
        manager: Job manager owning the engine lock.
    """
    with manager.engine_access():
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
def get_stats(
    engine: EchoVector = Depends(get_engine),
    manager: IndexJobManager = Depends(get_job_manager),
) -> StatsResponse:
    """Return index statistics.

    Args:
        engine: Engine to inspect.
        manager: Job manager owning the engine lock.
    """
    with manager.engine_access():
        s = engine.stats()
    return StatsResponse(
        chunks=int(s["chunks"]),
        embedding_dim=int(s["embedding_dim"]),
        store_dir=str(s["store_dir"]),
    )


@app.post("/reset", response_model=ResetResponse, dependencies=[Depends(require_token)])
def reset_index(
    engine: EchoVector = Depends(get_engine),
    manager: IndexJobManager = Depends(get_job_manager),
) -> ResetResponse:
    """Clear the index.

    Args:
        engine: Engine whose store is deleted.
        manager: Job manager owning the engine lock.

    Returns:
        The number of chunks that were present before the reset.

    Raises:
        HTTPException: 409 if an index job is in flight. Reset replaces the FAISS
            index and closes the metadata store, which a running job still holds.
    """
    active = manager.active_job_id
    if active is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Index job {active} is running; wait for it to finish before resetting.",
        )

    with manager.engine_access():
        chunks_removed = int(engine.stats()["chunks"])
        engine.reset()
    return ResetResponse(status="ok", chunks_removed=chunks_removed)
