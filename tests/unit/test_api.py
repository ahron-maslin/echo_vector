"""Unit tests for the EchoVector FastAPI server."""

import numpy as np
import pytest
import soundfile as sf

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx2")

from fastapi.testclient import TestClient  # noqa: E402

from echovector.api.server import app, get_engine  # noqa: E402


@pytest.fixture
def client(tmp_path):
    """TestClient wired to a fresh EchoVector index in tmp_path."""
    from echovector import EchoVector

    engine = EchoVector(
        store_dir=tmp_path / "idx",
        backend="local",
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        sample_rate=16_000,
    )
    app.dependency_overrides[get_engine] = lambda: engine
    yield TestClient(app)
    app.dependency_overrides.clear()


def _write_tone(path, frequency=440.0, sr=16_000):
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(path, (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sr)


def test_index_endpoint(client, tmp_path):
    """POST /index should index a file and report chunks_added > 0."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)

    resp = client.post("/index", json={"paths": [str(audio)]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks_added"] == 1
    assert body["files_skipped"] == 0


def test_index_skips_duplicate(client, tmp_path):
    """Indexing the same file twice should skip on the second call."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)

    client.post("/index", json={"paths": [str(audio)]})
    resp = client.post("/index", json={"paths": [str(audio)]})
    assert resp.status_code == 200
    assert resp.json()["chunks_added"] == 0
    assert resp.json()["files_skipped"] == 1


def test_search_endpoint(client, tmp_path):
    """POST /search should return results after indexing."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio, frequency=880.0)
    client.post("/index", json={"paths": [str(audio)]})

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
    client.post("/index", json={"paths": [str(audio)]})

    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks"] == 1
    assert body["embedding_dim"] > 0


def test_reset_endpoint(client, tmp_path):
    """POST /reset should clear the index."""
    audio = tmp_path / "tone.wav"
    _write_tone(audio)
    client.post("/index", json={"paths": [str(audio)]})

    resp = client.post("/reset")
    assert resp.status_code == 200

    stats = client.get("/stats").json()
    assert stats["chunks"] == 0
