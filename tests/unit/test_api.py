"""Unit tests for the EchoVector FastAPI server."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx2")

from fastapi.testclient import TestClient  # noqa: E402

from echovector.api.server import (  # noqa: E402
    TOKEN_ENV_VAR,
    IndexJobManager,
    app,
    get_engine,
    get_job_manager,
)

# Bounded poll for background jobs: never a bare sleep(), always a deadline.
JOB_TIMEOUT_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 0.01


@pytest.fixture
def manager():
    """A job manager with its own worker thread, shut down after the test."""
    mgr = IndexJobManager()
    yield mgr
    mgr.shutdown(wait=True)


@pytest.fixture
def engine(tmp_path):
    """A fresh local-backend EchoVector index in tmp_path."""
    from echovector import EchoVector

    return EchoVector(
        store_dir=tmp_path / "idx",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=16_000,
    )


@pytest.fixture
def client(engine, manager):
    """TestClient wired to a fresh EchoVector index and job manager."""
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_job_manager] = lambda: manager
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _no_token(monkeypatch):
    """Keep auth disabled unless a test opts in."""
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)


def _write_tone(path, frequency=440.0, sr=16_000):
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(path, (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sr)


def _await_job(client, job_id, headers=None, timeout=JOB_TIMEOUT_SECONDS):
    """Poll GET /jobs/{job_id} until it is terminal, or fail on a deadline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=headers).json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"Job {job_id} did not finish within {timeout}s")


def _index_now(client, paths, force=False):
    """Index synchronously via ?wait=true and return the finished job body."""
    resp = client.post("/index?wait=true", json={"paths": [str(p) for p in paths], "force": force})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Indexing — synchronous convenience path
# ---------------------------------------------------------------------------


def test_index_endpoint(client, tmp_path):
    """POST /index?wait=true should index a file and report chunks_added > 0."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)

    body = _index_now(client, [audio])
    assert body["status"] == "completed"
    assert body["chunks_added"] == 1
    assert body["files_skipped"] == 0


def test_index_skips_duplicate(client, tmp_path):
    """Indexing the same file twice should skip on the second call."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)

    _index_now(client, [audio])
    body = _index_now(client, [audio])
    assert body["chunks_added"] == 0
    assert body["files_skipped"] == 1


def test_index_skipped_count_with_mixed_paths(client, tmp_path):
    """files_skipped must reflect per-file status, not just whether any chunk was added."""
    already_indexed = tmp_path / "old.wav"
    _write_tone(already_indexed, frequency=220.0)
    _index_now(client, [already_indexed])

    new_file = tmp_path / "new.wav"
    _write_tone(new_file, frequency=660.0)

    body = _index_now(client, [already_indexed, new_file])
    assert body["chunks_added"] == 1
    assert body["files_skipped"] == 1


# ---------------------------------------------------------------------------
# Indexing — background jobs
# ---------------------------------------------------------------------------


def test_index_returns_immediately_with_a_job(client, tmp_path):
    """POST /index should default to 202 with a queued/running job id."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)

    resp = client.post("/index", json={"paths": [str(audio)]})
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"]
    assert body["status"] in {"queued", "running", "completed"}
    assert body["paths"] == [str(audio)]

    _await_job(client, body["job_id"])


def test_background_job_runs_to_completion(client, tmp_path):
    """A backgrounded job should finish, report progress, and add chunks."""
    for i, freq in enumerate((330.0, 550.0)):
        _write_tone(tmp_path / f"tone_{i}.wav", frequency=freq)

    resp = client.post("/index", json={"paths": [str(tmp_path)]})
    job = _await_job(client, resp.json()["job_id"])

    assert job["status"] == "completed"
    assert job["error"] is None
    assert job["files_total"] == 2
    assert job["files_done"] == 2
    assert job["progress"] == 1.0
    assert job["chunks_added"] == 2
    assert job["current_file"] is None
    assert job["started_at"] is not None
    assert job["finished_at"] is not None

    assert client.get("/stats").json()["chunks"] == 2


def test_failing_job_surfaces_its_error(client, tmp_path):
    """A job that raises should end as failed with the exception on the record."""
    missing = tmp_path / "does_not_exist.wav"

    resp = client.post("/index", json={"paths": [str(missing)]})
    assert resp.status_code == 202
    job = _await_job(client, resp.json()["job_id"])

    assert job["status"] == "failed"
    assert job["error"]
    assert "does_not_exist.wav" in job["error"]
    assert job["finished_at"] is not None


def test_list_jobs_is_most_recent_first(client, tmp_path):
    """GET /jobs should list every retained job, newest first."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)

    first = client.post("/index", json={"paths": [str(audio)]}).json()["job_id"]
    _await_job(client, first)
    second = client.post("/index", json={"paths": [str(audio)]}).json()["job_id"]
    _await_job(client, second)

    jobs = client.get("/jobs").json()["jobs"]
    assert [j["job_id"] for j in jobs] == [second, first]


def test_unknown_job_is_404(client):
    """GET /jobs/{job_id} should 404 for an id the server has never seen."""
    resp = client.get("/jobs/nope")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class _BlockingEngine:
    """Engine stub whose index() blocks until the test releases it."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def resolve_audio_files(self, paths, recursive=None):
        return [Path(p) for p in paths]

    def is_indexed(self, file_path):
        return False

    def index(self, targets, force=False, on_file_indexed=None, **kwargs):
        self.started.set()
        assert self.release.wait(JOB_TIMEOUT_SECONDS), "test never released the blocking engine"
        if on_file_indexed is not None:
            on_file_indexed(1, 1, targets[0])
        return 7

    def stats(self):
        return {"chunks": 0, "embedding_dim": 4, "store_dir": "memory://blocking"}

    def search(self, query, top_k=5):
        return []


@pytest.fixture
def blocking_client(manager):
    """TestClient backed by an engine whose indexing the test controls."""
    blocking = _BlockingEngine()
    app.dependency_overrides[get_engine] = lambda: blocking
    app.dependency_overrides[get_job_manager] = lambda: manager
    client = TestClient(app)
    try:
        yield client, blocking
    finally:
        blocking.release.set()
        app.dependency_overrides.clear()


def test_concurrent_index_is_rejected(blocking_client):
    """A second index request while one is running must be rejected with 409."""
    client, blocking = blocking_client

    first = client.post("/index", json={"paths": ["/a.wav"]}).json()
    assert blocking.started.wait(JOB_TIMEOUT_SECONDS)

    second = client.post("/index", json={"paths": ["/b.wav"]})
    assert second.status_code == 409
    detail = second.json()["detail"]
    assert first["job_id"] in detail
    assert "only one writer" in detail

    blocking.release.set()
    _await_job(client, first["job_id"])


def test_reset_is_rejected_while_indexing(blocking_client):
    """POST /reset must not run while a job holds the engine."""
    client, blocking = blocking_client

    job = client.post("/index", json={"paths": ["/a.wav"]}).json()
    assert blocking.started.wait(JOB_TIMEOUT_SECONDS)

    resp = client.post("/reset")
    assert resp.status_code == 409
    assert job["job_id"] in resp.json()["detail"]

    blocking.release.set()
    _await_job(client, job["job_id"])


def test_reads_are_served_during_indexing(blocking_client):
    """/search and /stats issued while a job is in flight must still be served.

    They are serialised behind the engine lock rather than run concurrently, so
    the read is issued from a second thread and only completes once the worker
    releases the file it is on.
    """
    client, blocking = blocking_client

    job = client.post("/index", json={"paths": ["/a.wav"]}).json()
    assert blocking.started.wait(JOB_TIMEOUT_SECONDS)

    statuses = {}

    def read():
        statuses["search"] = client.post("/search", json={"query": "anything"}).status_code
        statuses["stats"] = client.get("/stats").status_code

    reader = threading.Thread(target=read)
    reader.start()
    blocking.release.set()
    reader.join(JOB_TIMEOUT_SECONDS)

    assert not reader.is_alive()
    assert statuses == {"search": 200, "stats": 200}
    _await_job(client, job["job_id"])


# ---------------------------------------------------------------------------
# Search / stats / reset
# ---------------------------------------------------------------------------


def test_search_endpoint(client, tmp_path):
    """POST /search should return results after indexing."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio, frequency=880.0)
    _index_now(client, [audio])

    resp = client.post("/search", json={"query": "high tone", "top_k": 1})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert "tone.wav" in results[0]["filepath"]
    assert results[0]["score"] > 0.0


def test_search_empty_index(client):
    """POST /search on an empty index should return an empty list."""
    resp = client.post("/search", json={"query": "anything", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_stats_endpoint(client, tmp_path):
    """GET /stats should reflect the indexed chunk count."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)
    _index_now(client, [audio])

    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks"] == 1
    assert body["embedding_dim"] > 0


def test_reset_endpoint(client, tmp_path):
    """POST /reset should clear the index and report what it removed."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)
    _index_now(client, [audio])

    resp = client.post("/reset")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "chunks_removed": 1}

    assert client.get("/stats").json()["chunks"] == 0


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_without_engine():
    """GET /health must answer even with no engine configured and no overrides."""
    app.dependency_overrides.clear()
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["engine_configured"] is False
    assert body["auth_required"] is False


def test_health_reports_auth_and_active_job(client, monkeypatch):
    """GET /health should report the auth posture without needing a token."""
    monkeypatch.setenv(TOKEN_ENV_VAR, "s3cret")
    body = client.get("/health").json()
    assert body["auth_required"] is True
    assert body["active_job_id"] is None


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_mutating_routes_open_without_token(client, tmp_path):
    """With ECHOVECTOR_API_TOKEN unset the server keeps its open local behaviour."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)

    assert client.post("/index", json={"paths": [str(audio)]}).status_code == 202
    assert client.get("/jobs").status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [("post", "/index"), ("post", "/reset"), ("get", "/jobs")],
)
def test_protected_routes_reject_bad_token(client, monkeypatch, method, path):
    """Mutating and job routes must 401 without a matching bearer token."""
    monkeypatch.setenv(TOKEN_ENV_VAR, "s3cret")
    call = getattr(client, method)
    kwargs = {"json": {"paths": []}} if path == "/index" else {}

    assert call(path, **kwargs).status_code == 401
    assert call(path, headers={"Authorization": "Bearer wrong"}, **kwargs).status_code == 401
    assert call(path, headers={"Authorization": "s3cret"}, **kwargs).status_code == 401


def test_protected_routes_accept_matching_token(client, monkeypatch, tmp_path):
    """A matching bearer token unlocks the protected routes."""
    monkeypatch.setenv(TOKEN_ENV_VAR, "s3cret")
    headers = {"Authorization": "Bearer s3cret"}
    audio = tmp_path / "tone.wav"
    _write_tone(audio)

    resp = client.post("/index", json={"paths": [str(audio)]}, headers=headers)
    assert resp.status_code == 202
    job = _await_job(client, resp.json()["job_id"], headers=headers)
    assert job["status"] == "completed"

    assert client.get("/jobs", headers=headers).status_code == 200
    assert client.post("/reset", headers=headers).status_code == 200


def test_reads_stay_open_when_token_is_set(client, monkeypatch):
    """/search, /stats and /health are never token-gated."""
    monkeypatch.setenv(TOKEN_ENV_VAR, "s3cret")

    assert client.get("/health").status_code == 200
    assert client.get("/stats").status_code == 200
    assert client.post("/search", json={"query": "anything"}).status_code == 200
