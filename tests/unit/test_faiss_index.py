"""Unit tests for the Faiss index and SQLite store."""

import os
import tempfile
from collections.abc import Generator

import numpy as np
import pytest

from echovector.indexing import INDEX_TYPES, FaissIndex, SQLiteStore

ANN_TYPES = ("ivf", "hnsw")

#: Small enough to build quickly, large enough for IVF to have something to
#: cluster at the tuning defaults used below.
RECALL_CORPUS = 600
RECALL_DIM = 32


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


@pytest.mark.parametrize(
    "filepath",
    [
        "/audio/song_1.wav",  # `_` is a LIKE wildcard
        "/audio/song%1.wav",  # `%` is a LIKE wildcard
        "/audio/track#2.wav",  # `#` is the chunk-id separator
    ],
)
def test_filepath_matching_is_exact(store: SQLiteStore, filepath: str) -> None:
    """Chunks resolve to their source file by column, not by string pattern.

    Matching on a `LIKE 'path#%'` prefix mis-fires on filenames containing SQL
    wildcards or the separator itself.
    """
    store.add([0], [f"{filepath}#0.000-1.000"], [{"filepath": filepath}])

    assert store.has_filepath(filepath) is True
    assert store.get_int_ids_for_filepath(filepath) == [0]
    assert store.has_filepath("/audio/songX1.wav") is False


def test_filepath_column_backfills_for_legacy_stores(tmp_path) -> None:
    """Stores written before the filepath column must migrate on open."""
    import json
    import sqlite3

    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE metadata (int_id INTEGER PRIMARY KEY, "
        "string_id TEXT UNIQUE NOT NULL, metadata_json TEXT)"
    )
    conn.execute(
        "INSERT INTO metadata VALUES (?, ?, ?)",
        (0, "/audio/old.wav#0.000-1.000", json.dumps({"filepath": "/audio/old.wav"})),
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(str(db_path))
    try:
        assert store.has_filepath("/audio/old.wav") is True
        assert store.count_files() == 1
    finally:
        store.close()


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


# ----------------------------------------------------------------------
# File change detection
# ----------------------------------------------------------------------


def test_file_signature_round_trip(store: SQLiteStore) -> None:
    """mtime and size promoted out of the metadata dict come back intact."""
    store.add(
        [0],
        ["/audio/a.wav#0.000-1.000"],
        [{"filepath": "/audio/a.wav", "file_mtime": 1700000000.5, "file_size": 4096}],
    )

    assert store.get_file_signature("/audio/a.wav") == (1700000000.5, 4096)


def test_file_signature_is_none_when_unknown(store: SQLiteStore) -> None:
    """An unindexed file, or one indexed before signatures existed, reads as unknown."""
    store.add([0], ["/audio/a.wav#0.000-1.000"], [{"filepath": "/audio/a.wav"}])

    assert store.get_file_signature("/audio/a.wav") is None
    assert store.get_file_signature("/audio/never-indexed.wav") is None


def test_list_filepaths_is_distinct_and_sorted(store: SQLiteStore) -> None:
    store.add(
        [0, 1, 2],
        ["b#0", "b#1", "a#0"],
        [{"filepath": "/audio/b.wav"}, {"filepath": "/audio/b.wav"}, {"filepath": "/audio/a.wav"}],
    )

    assert store.list_filepaths() == ["/audio/a.wav", "/audio/b.wav"]


def test_signature_columns_backfill_for_legacy_stores(tmp_path) -> None:
    """Stores written before the signature columns migrate and backfill on open."""
    import json
    import sqlite3

    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE metadata (int_id INTEGER PRIMARY KEY, "
        "string_id TEXT UNIQUE NOT NULL, filepath TEXT, metadata_json TEXT)"
    )
    conn.execute(
        "INSERT INTO metadata VALUES (?, ?, ?, ?)",
        (
            0,
            "/audio/old.wav#0.000-1.000",
            "/audio/old.wav",
            json.dumps({"filepath": "/audio/old.wav", "file_size": 128}),
        ),
    )
    conn.execute(
        "INSERT INTO metadata VALUES (?, ?, ?, ?)",
        (1, "/audio/bare.wav#0.000-1.000", "/audio/bare.wav", json.dumps({})),
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(str(db_path))
    try:
        # file_size backfills from the stored blob, but mtime was never
        # recorded, so the signature stays unknown and the file re-indexes.
        assert store.get_file_signature("/audio/old.wav") is None
        assert store.get_file_signature("/audio/bare.wav") is None
        assert store.list_filepaths() == ["/audio/bare.wav", "/audio/old.wav"]

        store.add(
            [0],
            ["/audio/old.wav#0.000-1.000"],
            [{"filepath": "/audio/old.wav", "file_mtime": 12.0, "file_size": 128}],
        )
        assert store.get_file_signature("/audio/old.wav") == (12.0, 128)
    finally:
        store.close()


# ----------------------------------------------------------------------
# Approximate index types
# ----------------------------------------------------------------------


def _clustered(count: int, dimension: int, seed: int = 11) -> np.ndarray:
    """Build normalized vectors with cluster structure, as audio embeddings have."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((16, dimension)).astype(np.float32) * 4.0
    points = centers[rng.integers(0, 16, count)] + rng.standard_normal((count, dimension)).astype(
        np.float32
    )
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return np.ascontiguousarray(points, dtype=np.float32)


def _fill(index: FaissIndex, vectors: np.ndarray) -> None:
    index.add(
        vectors, [f"c{i}" for i in range(len(vectors))], [{"n": i} for i in range(len(vectors))]
    )


def test_unknown_index_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown index_type"):
        FaissIndex(dimension=4, index_type="annoy")


@pytest.mark.parametrize("index_type", INDEX_TYPES)
def test_index_types_round_trip_through_disk(tmp_path, index_type: str) -> None:
    """Every structure must save, reload, and still resolve metadata by ID."""
    vectors = _clustered(200, 8)
    db_path = str(tmp_path / "meta.sqlite")
    index_path = str(tmp_path / "index.faiss")

    index = FaissIndex(dimension=8, db_path=db_path, index_type=index_type, nlist=8)
    _fill(index, vectors)
    index.save(index_path)
    index.store.close()

    # A fresh instance defaults to "flat"; the file on disk must win.
    reloaded = FaissIndex(dimension=8, db_path=db_path)
    try:
        reloaded.load(index_path)
        assert reloaded.index.ntotal == 200
        assert reloaded.index_type == index_type
        assert reloaded.desync_report() is None

        _, ids, meta = reloaded.search(vectors[:1], k=1)
        assert ids[0][0] is not None
        assert meta[0][0] is not None
    finally:
        reloaded.store.close()


def test_ivf_rejects_a_first_batch_too_small_to_train() -> None:
    """Faiss would assert deep inside clustering; say what to do instead."""
    index = FaissIndex(dimension=8, index_type="ivf", nlist=32)

    with pytest.raises(ValueError, match="at least nlist=32"):
        index.add(_clustered(10, 8), [f"c{i}" for i in range(10)])

    # The rejected batch must leave both halves untouched.
    assert index.index.ntotal == 0
    assert index.store.count() == 0


def test_ivf_trains_on_the_first_batch_then_accepts_small_ones() -> None:
    index = FaissIndex(dimension=8, index_type="ivf", nlist=8)
    _fill(index, _clustered(64, 8))
    assert index.index.is_trained
    assert index.index.ntotal == 64

    index.add(_clustered(2, 8, seed=99), ["late-a", "late-b"], [{}, {}])
    assert index.index.ntotal == 66
    assert index.desync_report() is None


def test_removal_support_by_index_type() -> None:
    """HNSW is the one structure Faiss cannot delete from in place."""
    assert FaissIndex(dimension=4).supports_native_removal() is True
    assert FaissIndex(dimension=4, index_type="ivf").supports_native_removal() is True
    assert FaissIndex(dimension=4, index_type="hnsw").supports_native_removal() is False


@pytest.mark.parametrize("index_type", INDEX_TYPES)
def test_remove_int_ids_works_for_every_index_type(index_type: str) -> None:
    """Removal must clear both halves whether Faiss supports it natively or not."""
    vectors = _clustered(40, 8)
    index = FaissIndex(dimension=8, index_type=index_type, nlist=4)
    _fill(index, vectors)

    doomed = index.store.get_int_ids_for_string_ids(["c3", "c7"])
    index.remove_int_ids(doomed)

    assert index.index.ntotal == 38
    assert index.store.count() == 38
    assert index.desync_report() is None
    import faiss

    surviving = {int(i) for i in faiss.vector_to_array(index.index.id_map)}
    assert not surviving & set(doomed)
    # IDs are never recycled, even when a rebuild reconstructed the index.
    assert index.next_int_id() == 40


@pytest.mark.parametrize("index_type", ANN_TYPES)
def test_ann_index_approximates_the_flat_top_results(index_type: str) -> None:
    """Tuning defaults must actually retrieve; a fast index that misses is useless."""
    vectors = _clustered(RECALL_CORPUS, RECALL_DIM)
    queries = _clustered(25, RECALL_DIM, seed=5)
    k = 10

    exact = FaissIndex(dimension=RECALL_DIM)
    _fill(exact, vectors)
    _, exact_ids, _ = exact.search(queries, k=k)

    approximate = FaissIndex(dimension=RECALL_DIM, index_type=index_type)
    _fill(approximate, vectors)
    _, approximate_ids, _ = approximate.search(queries, k=k)

    overlap = [len(set(a) & set(b)) / k for a, b in zip(exact_ids, approximate_ids, strict=True)]
    recall = sum(overlap) / len(overlap)

    assert recall >= 0.8, f"{index_type} recall@{k} was {recall:.2f} at the default tuning"
