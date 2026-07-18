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


def test_has_filepath_does_not_match_like_wildcards(store: SQLiteStore) -> None:
    """Filenames containing literal `_`/`%` must not act as SQL LIKE wildcards."""
    store.add([0], ["/audio/songX1.wav#0.000-1.000"], [{}])

    assert store.has_filepath("/audio/song_1.wav") is False
    assert store.get_int_ids_for_filepath("/audio/song_1.wav") == []
    assert store.has_filepath("/audio/songX1.wav") is True


def test_sqlite_store_add_mismatched_lengths(store: SQLiteStore) -> None:
    with pytest.raises(ValueError, match="Mismatched lengths"):
        store.add([0, 1], ["only-one"], [{}])


def test_sqlite_store_get_by_int_ids_empty(store: SQLiteStore) -> None:
    assert store.get_by_int_ids([]) == ([], [])


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


def test_faiss_index_add_validation_errors() -> None:
    """Test that add() rejects malformed inputs."""
    index = FaissIndex(dimension=4)

    with pytest.raises(ValueError, match="dimension"):
        index.add(np.zeros((1, 3), dtype=np.float32), ["a"])

    with pytest.raises(ValueError, match="Number of IDs"):
        index.add(np.zeros((2, 4), dtype=np.float32), ["a"])

    with pytest.raises(ValueError, match="Length of metadata"):
        index.add(np.zeros((2, 4), dtype=np.float32), ["a", "b"], [{}])


def test_faiss_index_search_validation_error() -> None:
    """Test that search() rejects malformed query embeddings."""
    index = FaissIndex(dimension=4)
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.zeros((1, 3), dtype=np.float32), k=1)


def test_faiss_index_remove_int_ids() -> None:
    """Test removing vectors by integer ID."""
    index = FaissIndex(dimension=4)
    index.add(
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        ["a", "b"],
        [{}, {}],
    )
    assert index.index.ntotal == 2

    index.remove_int_ids([])  # no-op
    assert index.index.ntotal == 2

    index.remove_int_ids([0])
    assert index.index.ntotal == 1
    str_ids, _ = index.store.get_by_int_ids([0])
    assert str_ids == [None]


def test_faiss_index_load_missing_file() -> None:
    """Test that load() raises FileNotFoundError for a missing index file."""
    index = FaissIndex(dimension=4)
    with pytest.raises(FileNotFoundError):
        index.load("/nonexistent/path/index.faiss")


def test_faiss_index_load_dimension_mismatch() -> None:
    """Test that load() rejects an index with a different dimension."""
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "test.index")
        FaissIndex(dimension=4).save(index_path)

        mismatched = FaissIndex(dimension=8)
        with pytest.raises(ValueError, match="dimension"):
            mismatched.load(index_path)


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
