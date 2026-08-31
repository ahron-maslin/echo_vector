"""SQLite-based store for metadata persistence."""

import contextlib
import json
import sqlite3
from typing import Any

from .base import BaseStore


class SQLiteStore(BaseStore):
    """SQLite-based store for metadata and string ID persistence."""

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize the SQLite store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.initialize()

    def initialize(self) -> None:
        """Initialize the database schema, migrating older stores in place."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                int_id INTEGER PRIMARY KEY,
                string_id TEXT UNIQUE NOT NULL,
                filepath TEXT,
                file_mtime REAL,
                file_size INTEGER,
                metadata_json TEXT
            )
            """
        )
        self._migrate_add_filepath_column(cursor)
        self._migrate_add_file_signature_columns(cursor)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_filepath ON metadata(filepath)")
        self._conn.commit()

    def _migrate_add_filepath_column(self, cursor: sqlite3.Cursor) -> None:
        """Add and backfill `filepath` for stores written before it existed.

        Earlier versions matched chunks to source files with a `LIKE 'path#%'`
        prefix query, which mis-matches whenever a filename itself contains a
        `#`. The column makes the association exact.

        Args:
            cursor: Cursor on the store's connection; not committed here.
        """
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(metadata)")}
        if "filepath" in columns:
            return

        cursor.execute("ALTER TABLE metadata ADD COLUMN filepath TEXT")
        rows = cursor.execute("SELECT int_id, metadata_json FROM metadata").fetchall()
        updates = [
            (self._as_str(self._metadata_field(metadata_json, "filepath")), int_id)
            for int_id, metadata_json in rows
        ]
        if updates:
            cursor.executemany("UPDATE metadata SET filepath = ? WHERE int_id = ?", updates)

    def _migrate_add_file_signature_columns(self, cursor: sqlite3.Cursor) -> None:
        """Add and backfill `file_mtime` / `file_size` for older stores.

        Without a signature a source file is recognised by path alone, so an
        edited or replaced recording is never re-indexed. Backfilled from the
        stored metadata blob where it already carries the values; rows that do
        not are left NULL, which reads as "signature unknown" and makes
        :meth:`get_file_signature` return None so the caller re-indexes.

        Args:
            cursor: Cursor on the store's connection; not committed here.
        """
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(metadata)")}
        missing = [name for name in ("file_mtime", "file_size") if name not in columns]
        if not missing:
            return

        types = {"file_mtime": "REAL", "file_size": "INTEGER"}
        for name in missing:
            cursor.execute(f"ALTER TABLE metadata ADD COLUMN {name} {types[name]}")

        rows = cursor.execute("SELECT int_id, metadata_json FROM metadata").fetchall()
        if not rows:
            return

        assignments = ", ".join(f"{name} = ?" for name in missing)
        coerce = {"file_mtime": self._as_float, "file_size": self._as_int}
        updates = [
            (
                *(coerce[name](self._metadata_field(metadata_json, name)) for name in missing),
                int_id,
            )
            for int_id, metadata_json in rows
        ]
        cursor.executemany(
            f"UPDATE metadata SET {assignments} WHERE int_id = ?",  # noqa: S608
            updates,
        )

    @staticmethod
    def _metadata_field(metadata_json: str | None, key: str) -> Any:
        """Read one key out of a stored metadata blob.

        Args:
            metadata_json: The serialized metadata, possibly absent or corrupt.
            key: The key to read.

        Returns:
            The value, or None if it is absent or the blob is unusable.
        """
        if not metadata_json:
            return None
        try:
            return json.loads(metadata_json).get(key)
        except (json.JSONDecodeError, AttributeError):
            return None

    @staticmethod
    def _as_str(value: Any) -> str | None:
        """Coerce a metadata value to a string, preserving None.

        Args:
            value: The raw value from a metadata dictionary.

        Returns:
            The string, or None.
        """
        return None if value is None else str(value)

    @staticmethod
    def _as_float(value: Any) -> float | None:
        """Coerce a metadata value to a float, or None if it is not numeric.

        Args:
            value: The raw value from a metadata dictionary.

        Returns:
            The float, or None.
        """
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        """Coerce a metadata value to an int, or None if it is not numeric.

        Args:
            value: The raw value from a metadata dictionary.

        Returns:
            The integer, or None.
        """
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def add(
        self, int_ids: list[int], string_ids: list[str], metadata_list: list[dict[str, Any]]
    ) -> None:
        """Add metadata and ID mappings to the store.

        The source file's modification time and size are promoted out of the
        metadata dictionary into their own columns when present, so a later
        indexing run can tell an unchanged file from an edited one. The store
        never stats the file itself; whatever the caller recorded at index time
        is what gets written.

        Args:
            int_ids: List of integer IDs assigned by the index.
            string_ids: List of original string IDs.
            metadata_list: List of metadata dictionaries.

        Raises:
            ValueError: If lengths of input lists do not match.
        """
        if not (len(int_ids) == len(string_ids) == len(metadata_list)):
            raise ValueError("Mismatched lengths for IDs and metadata.")

        cursor = self._conn.cursor()
        data = [
            (
                i_id,
                s_id,
                self._as_str(meta.get("filepath")),
                self._as_float(meta.get("file_mtime")),
                self._as_int(meta.get("file_size")),
                json.dumps(meta),
            )
            for i_id, s_id, meta in zip(int_ids, string_ids, metadata_list, strict=True)
        ]

        cursor.executemany(
            """
            INSERT OR REPLACE INTO metadata
                (int_id, string_id, filepath, file_mtime, file_size, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            data,
        )
        self._conn.commit()

    def get_by_int_ids(
        self, int_ids: list[int]
    ) -> tuple[list[str | None], list[dict[str, Any] | None]]:
        """Retrieve string IDs and metadata for a list of integer IDs.

        Args:
            int_ids: List of integer IDs to query.

        Returns:
            A tuple containing a list of string IDs and a list of metadata dictionaries.
        """
        if not int_ids:
            return [], []

        cursor = self._conn.cursor()
        placeholders = ",".join("?" for _ in int_ids)
        cols = "int_id, string_id, metadata_json"
        query = f"SELECT {cols} FROM metadata WHERE int_id IN ({placeholders})"  # noqa: S608
        cursor.execute(query, int_ids)

        rows = cursor.fetchall()
        row_dict = {row[0]: (row[1], json.loads(row[2])) for row in rows}

        string_ids: list[str | None] = []
        metadata: list[dict[str, Any] | None] = []
        for i_id in int_ids:
            if i_id in row_dict:
                string_ids.append(row_dict[i_id][0])
                metadata.append(row_dict[i_id][1])
            else:
                string_ids.append(None)
                metadata.append(None)

        return string_ids, metadata

    def get_max_int_id(self) -> int:
        """Get the maximum integer ID currently in the store.

        Returns:
            The maximum integer ID, or -1 if empty.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT MAX(int_id) FROM metadata")
        row = cursor.fetchone()
        return int(row[0]) if row[0] is not None else -1

    def count(self) -> int:
        """Return the number of chunk rows currently in the store.

        Returns:
            The row count.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM metadata")
        return int(cursor.fetchone()[0])

    def count_files(self) -> int:
        """Return the number of distinct source files represented in the store.

        Returns:
            The distinct filepath count.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT filepath) FROM metadata WHERE filepath IS NOT NULL")
        return int(cursor.fetchone()[0])

    def get_int_ids_for_string_ids(self, string_ids: list[str]) -> list[int]:
        """Return the integer IDs currently mapped to the given string IDs.

        Used to evict superseded vectors before re-adding the same string ID,
        so the vector index never keeps rows the store has dropped.

        Args:
            string_ids: String IDs to look up.

        Returns:
            Integer IDs for whichever string IDs are present.
        """
        if not string_ids:
            return []
        cursor = self._conn.cursor()
        placeholders = ",".join("?" for _ in string_ids)
        query = f"SELECT int_id FROM metadata WHERE string_id IN ({placeholders})"  # noqa: S608
        cursor.execute(query, string_ids)
        return [int(row[0]) for row in cursor.fetchall()]

    def has_filepath(self, filepath: str) -> bool:
        """Return True if any chunk from filepath is already stored.

        Args:
            filepath: Path of the source audio file, as recorded at index time.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM metadata WHERE filepath = ? LIMIT 1", (filepath,))
        return cursor.fetchone() is not None

    def get_file_signature(self, filepath: str) -> tuple[float, int] | None:
        """Return the recorded (mtime, size) of an indexed source file.

        Compare against the file on disk to decide whether it must be
        re-indexed. None means the store cannot vouch for the file — it was
        never indexed, or it predates the signature columns — in which case the
        caller should treat it as changed rather than assume it is current.

        Args:
            filepath: Path of the source audio file, as recorded at index time.

        Returns:
            The (modification time, size in bytes) pair, or None if unknown.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT file_mtime, file_size FROM metadata "
            "WHERE filepath = ? AND file_mtime IS NOT NULL AND file_size IS NOT NULL "
            "LIMIT 1",
            (filepath,),
        )
        row = cursor.fetchone()
        return (float(row[0]), int(row[1])) if row is not None else None

    def list_filepaths(self) -> list[str]:
        """Return every distinct source file represented in the store.

        Returns:
            Sorted filepaths, as recorded at index time.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT DISTINCT filepath FROM metadata WHERE filepath IS NOT NULL ORDER BY filepath"
        )
        return [str(row[0]) for row in cursor.fetchall()]

    def get_int_ids_for_filepath(self, filepath: str) -> list[int]:
        """Return all integer IDs belonging to chunks of filepath.

        Args:
            filepath: Source audio file path.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT int_id FROM metadata WHERE filepath = ?", (filepath,))
        return [int(row[0]) for row in cursor.fetchall()]

    def delete_by_filepath(self, filepath: str) -> list[int]:
        """Delete all chunks for filepath and return their integer IDs.

        Args:
            filepath: Source audio file path.

        Returns:
            List of integer IDs that were removed.
        """
        int_ids = self.get_int_ids_for_filepath(filepath)
        self.delete_by_int_ids(int_ids)
        return int_ids

    def delete_by_int_ids(self, int_ids: list[int]) -> None:
        """Delete rows by integer ID.

        Args:
            int_ids: Integer IDs to remove.
        """
        if not int_ids:
            return
        placeholders = ",".join("?" for _ in int_ids)
        query = f"DELETE FROM metadata WHERE int_id IN ({placeholders})"  # noqa: S608
        self._conn.execute(query, int_ids)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        """Ensure connection is closed on GC to avoid Python 3.13 sqlite3 finalizer bug."""
        with contextlib.suppress(Exception):
            self._conn.close()
