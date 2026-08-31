"""Regression tests for index/store consistency.

Each case here corresponds to a way the vector index and its metadata store
could silently drift apart, producing confident but wrong search results.
"""

import numpy as np
import pytest

from echovector.indexing import INDEX_TYPES, FaissIndex


@pytest.fixture
def index(tmp_path) -> FaissIndex:
    faiss_index = FaissIndex(dimension=4, db_path=str(tmp_path / "meta.sqlite"))
    yield faiss_index
    faiss_index.store.close()


def _vec(*values: float) -> np.ndarray:
    return np.array([values], dtype=np.float32)


def test_readding_a_string_id_evicts_the_old_vector(index: FaissIndex) -> None:
    """The store replaces on conflict; the vector index must not keep the orphan.

    An orphan stays searchable forever, occupies a result slot, and resolves to
    no metadata — so a top-k query quietly returns fewer than k rows.
    """
    index.add(_vec(1, 0, 0, 0), ["chunk-1"], [{"version": 1}])
    index.add(_vec(1, 0, 0, 0), ["chunk-1"], [{"version": 2}])

    assert index.index.ntotal == 1
    assert index.store.count() == 1
    assert index.desync_report() is None

    _, ids, metadata = index.search(_vec(1, 0, 0, 0), k=2)
    assert ids[0] == ["chunk-1"]
    assert metadata[0] == [{"version": 2}]


def test_full_result_page_after_repeated_reindexing(index: FaissIndex) -> None:
    """Re-indexing the same chunks must not erode how many results a query returns."""
    for _ in range(3):
        index.add(
            np.eye(4, dtype=np.float32),
            ["a", "b", "c", "d"],
            [{"n": i} for i in range(4)],
        )

    _, ids, _ = index.search(_vec(1, 0, 0, 0), k=4)

    assert index.index.ntotal == 4
    # Ties among the orthogonal vectors break arbitrarily; what matters is that
    # the page is full and every slot resolves to a real chunk.
    assert set(ids[0]) == {"a", "b", "c", "d"}


def test_next_int_id_survives_a_lost_metadata_store(tmp_path) -> None:
    """Losing the store must not restart IDs on top of ones Faiss still holds."""
    db_path = tmp_path / "meta.sqlite"
    index = FaissIndex(dimension=4, db_path=str(db_path))
    index.add(np.eye(4, dtype=np.float32)[:2], ["a", "b"], [{}, {}])
    index.save(str(tmp_path / "index.faiss"))
    index.store.close()

    db_path.unlink()
    recovered = FaissIndex(dimension=4, db_path=str(db_path))
    recovered.load(str(tmp_path / "index.faiss"))
    try:
        assert recovered.store.get_max_int_id() == -1
        assert recovered.max_faiss_int_id() == 1
        assert recovered.next_int_id() == 2

        recovered.add(_vec(0, 0, 1, 0), ["c"], [{}])
        ids = [int(i) for i in recovered.store.get_int_ids_for_string_ids(["c"])]
        assert ids == [2]
    finally:
        recovered.store.close()


def test_desync_report_describes_the_mismatch(tmp_path) -> None:
    db_path = tmp_path / "meta.sqlite"
    index = FaissIndex(dimension=4, db_path=str(db_path))
    index.add(np.eye(4, dtype=np.float32)[:2], ["a", "b"], [{}, {}])
    index.save(str(tmp_path / "index.faiss"))
    index.store.close()

    db_path.unlink()
    recovered = FaissIndex(dimension=4, db_path=str(db_path))
    recovered.load(str(tmp_path / "index.faiss"))
    try:
        report = recovered.desync_report()
        assert report is not None
        assert "2 vector(s)" in report
        assert "0 row(s)" in report
    finally:
        recovered.store.close()


def test_max_faiss_int_id_on_empty_index(index: FaissIndex) -> None:
    assert index.max_faiss_int_id() == -1
    assert index.next_int_id() == 0


def test_save_is_atomic(tmp_path, index: FaissIndex) -> None:
    """A save must not leave a partial file where the index used to be."""
    index_path = tmp_path / "index.faiss"
    index.add(np.eye(4, dtype=np.float32)[:1], ["a"], [{}])
    index.save(str(index_path))
    first_size = index_path.stat().st_size

    index.add(np.eye(4, dtype=np.float32)[1:3], ["b", "c"], [{}, {}])
    index.save(str(index_path))

    assert not (tmp_path / "index.faiss.tmp").exists()
    assert index_path.stat().st_size > first_size


def test_remove_int_ids_clears_both_halves(index: FaissIndex) -> None:
    index.add(np.eye(4, dtype=np.float32), ["a", "b", "c", "d"], [{}] * 4)
    to_remove = index.store.get_int_ids_for_string_ids(["b", "c"])

    index.remove_int_ids(to_remove)

    assert index.index.ntotal == 2
    assert index.store.count() == 2
    assert index.desync_report() is None


def test_remove_int_ids_with_empty_list_is_a_noop(index: FaissIndex) -> None:
    index.add(_vec(1, 0, 0, 0), ["a"], [{}])
    index.remove_int_ids([])
    assert index.index.ntotal == 1


def test_loading_an_index_without_an_id_map_is_refused(tmp_path) -> None:
    """A legacy index cannot join to metadata; say so instead of failing deep in Faiss."""
    import faiss

    legacy_path = tmp_path / "legacy.faiss"
    legacy = faiss.IndexFlatIP(4)
    legacy.add(np.eye(4, dtype=np.float32))
    faiss.write_index(legacy, str(legacy_path))

    index = FaissIndex(dimension=4, db_path=str(tmp_path / "meta.sqlite"))
    try:
        with pytest.raises(ValueError, match="no ID map"):
            index.load(str(legacy_path))
    finally:
        index.store.close()


def test_loading_a_mismatched_dimension_is_refused(tmp_path) -> None:
    index_path = tmp_path / "index.faiss"
    wide = FaissIndex(dimension=8, db_path=str(tmp_path / "wide.sqlite"))
    wide.add(np.eye(8, dtype=np.float32)[:1], ["a"], [{}])
    wide.save(str(index_path))
    wide.store.close()

    narrow = FaissIndex(dimension=4, db_path=str(tmp_path / "narrow.sqlite"))
    try:
        with pytest.raises(ValueError, match="does not match"):
            narrow.load(str(index_path))
    finally:
        narrow.store.close()


def test_loading_a_missing_index_raises(tmp_path) -> None:
    index = FaissIndex(dimension=4, db_path=str(tmp_path / "meta.sqlite"))
    try:
        with pytest.raises(FileNotFoundError):
            index.load(str(tmp_path / "nope.faiss"))
    finally:
        index.store.close()


# ----------------------------------------------------------------------
# The same invariants, for every approximate structure
# ----------------------------------------------------------------------


def _corpus(count: int, dimension: int = 8, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((count, dimension)).astype(np.float32)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return np.ascontiguousarray(points, dtype=np.float32)


@pytest.mark.parametrize("index_type", INDEX_TYPES)
def test_readding_a_string_id_evicts_the_old_vector_for_every_type(
    tmp_path, index_type: str
) -> None:
    """Approximate structures must not become orphan farms.

    HNSW cannot delete in place, so this also exercises the rebuild path that
    keeps the eviction invariant true there.
    """
    vectors = _corpus(32)
    index = FaissIndex(
        dimension=8,
        db_path=str(tmp_path / "meta.sqlite"),
        index_type=index_type,
        nlist=4,
    )
    try:
        ids = [f"c{i}" for i in range(32)]
        index.add(vectors, ids, [{"version": 1} for _ in ids])
        index.add(vectors, ids, [{"version": 2} for _ in ids])

        assert index.index.ntotal == 32
        assert index.store.count() == 32
        assert index.desync_report() is None

        _, found, metadata = index.search(vectors[:1], k=1)
        assert found[0][0] is not None
        assert metadata[0][0] == {"version": 2}
    finally:
        index.store.close()


@pytest.mark.parametrize("index_type", INDEX_TYPES)
def test_next_int_id_survives_a_lost_store_for_every_type(tmp_path, index_type: str) -> None:
    """IDs still come from both halves once the structure is reloaded from disk."""
    db_path = tmp_path / "meta.sqlite"
    index_path = tmp_path / "index.faiss"
    index = FaissIndex(dimension=8, db_path=str(db_path), index_type=index_type, nlist=4)
    index.add(_corpus(32), [f"c{i}" for i in range(32)], [{} for _ in range(32)])
    index.save(str(index_path))
    index.store.close()

    db_path.unlink()
    recovered = FaissIndex(dimension=8, db_path=str(db_path))
    recovered.load(str(index_path))
    try:
        assert recovered.store.get_max_int_id() == -1
        assert recovered.max_faiss_int_id() == 31
        assert recovered.next_int_id() == 32
    finally:
        recovered.store.close()
