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
        """Initialize the database schema."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                int_id INTEGER PRIMARY KEY,
                string_id TEXT UNIQUE NOT NULL,
                metadata_json TEXT
            )
            """
        )
        self._conn.commit()

    def add(
        self, int_ids: list[int], string_ids: list[str], metadata_list: list[dict[str, Any]]
    ) -> None:
        """Add metadata and ID mappings to the store.

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
            (i_id, s_id, json.dumps(meta))
            for i_id, s_id, meta in zip(int_ids, string_ids, metadata_list, strict=True)
        ]

        cursor.executemany(
            """
            INSERT OR REPLACE INTO metadata (int_id, string_id, metadata_json)
            VALUES (?, ?, ?)
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

    def has_filepath(self, filepath: str) -> bool:
        """Return True if any chunk from filepath is already stored.

        Args:
            filepath: Absolute or relative path of the source audio file.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT 1 FROM metadata WHERE string_id LIKE ? LIMIT 1",
            (filepath + "#%",),
        )
        return cursor.fetchone() is not None

    def get_int_ids_for_filepath(self, filepath: str) -> list[int]:
        """Return all integer IDs belonging to chunks of filepath.

        Args:
            filepath: Source audio file path.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT int_id FROM metadata WHERE string_id LIKE ?",
            (filepath + "#%",),
        )
        return [row[0] for row in cursor.fetchall()]

    def delete_by_filepath(self, filepath: str) -> list[int]:
        """Delete all chunks for filepath and return their integer IDs.

        Args:
            filepath: Source audio file path.

        Returns:
            List of integer IDs that were removed.
        """
        int_ids = self.get_int_ids_for_filepath(filepath)
        if int_ids:
            placeholders = ",".join("?" for _ in int_ids)
            delete_query = f"DELETE FROM metadata WHERE int_id IN ({placeholders})"  # noqa: S608
            self._conn.execute(delete_query, int_ids)
            self._conn.commit()
        return int_ids

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        """Ensure connection is closed on GC to avoid Python 3.13 sqlite3 finalizer bug."""
        with contextlib.suppress(Exception):
            self._conn.close()
