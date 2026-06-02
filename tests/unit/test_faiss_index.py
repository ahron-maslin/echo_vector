"""Unit tests for the Faiss index and SQLite store."""

import os
import tempfile
from collections.abc import Generator

import numpy as np
import pytest

from echovector.indexing import FaissIndex, SQLiteStore


@pytest.fixture
def store() -> Generator[SQLiteStore, None, None]:
    """Provide an in-memory SQLite store."""
    db = SQLiteStore(":memory:")
    yield db
    db.close()


def test_sqlite_store(store: SQLiteStore) -> None:
    """Test SQLiteStore basic operations."""
    int_ids = [0, 1]
    str_ids = ["doc1", "doc2"]
    metadata = [{"author": "alice"}, {"author": "bob"}]

    store.add(int_ids, str_ids, metadata)

    assert store.get_max_int_id() == 1

    ret_str, ret_meta = store.get_by_int_ids([0, 1, 99])
    assert ret_str == ["doc1", "doc2", None]
    assert ret_meta == [{"author": "alice"}, {"author": "bob"}, None]


def test_faiss_index_initialization() -> None:
    """Test FaissIndex initialization."""
    index = FaissIndex(dimension=128)
    assert index.dimension == 128
    assert index.index.ntotal == 0


def test_faiss_index_add_and_search() -> None:
    """Test batched and incremental indexing and searching."""
    index = FaissIndex(dimension=4)

    # First batch
    embeddings1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    ids1 = ["a", "b"]
    meta1 = [{"val": 1}, {"val": 2}]

    index.add(embeddings1, ids1, meta1)
    assert index.index.ntotal == 2

    # Second batch (incremental)
    embeddings2 = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    ids2 = ["c", "d"]
    meta2 = [{"val": 3}, {"val": 4}]

    index.add(embeddings2, ids2, meta2)
    assert index.index.ntotal == 4

    # Search
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    _dists, res_ids, res_meta = index.search(query, k=2)

    assert res_ids[0][0] == "a"
    assert res_meta[0][0] == {"val": 1}
    assert res_ids[0][1] in (
        "b",
        "c",
        "d",
        None,
    )  # Since inner product with orthogonal vectors is 0


def test_faiss_index_search_empty() -> None:
    """Test search on an empty index."""
    index = FaissIndex(dimension=4)
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    dists, res_ids, res_meta = index.search(query, k=2)

    assert dists.size == 0
    assert len(res_ids) == 1
    assert len(res_ids[0]) == 0
    assert len(res_meta) == 1
    assert len(res_meta[0]) == 0


def test_faiss_index_save_load() -> None:
    """Test saving and loading the index."""
    dimension = 4
    index = FaissIndex(dimension=dimension)

    embeddings = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    index.add(embeddings, ["test_id"], [{"test": "data"}])

    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "test.index")
        index.save(index_path)

        new_index = FaissIndex(dimension=dimension)
        new_index.load(index_path)

        assert new_index.index.ntotal == 1
