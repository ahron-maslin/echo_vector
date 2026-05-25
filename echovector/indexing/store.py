"""SQLite-based store for metadata persistence."""

import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

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
            '''
            CREATE TABLE IF NOT EXISTS metadata (
                int_id INTEGER PRIMARY KEY,
                string_id TEXT UNIQUE NOT NULL,
                metadata_json TEXT
            )
            '''
        )
        self._conn.commit()

    def add(
        self, int_ids: List[int], string_ids: List[str], metadata_list: List[Dict[str, Any]]
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
            '''
            INSERT OR REPLACE INTO metadata (int_id, string_id, metadata_json)
            VALUES (?, ?, ?)
            ''',
            data,
        )
        self._conn.commit()

    def get_by_int_ids(
        self, int_ids: List[int]
    ) -> Tuple[List[Optional[str]], List[Optional[Dict[str, Any]]]]:
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
        cursor.execute(
            f'''
            SELECT int_id, string_id, metadata_json
            FROM metadata
            WHERE int_id IN ({placeholders})
            ''',
            int_ids,
        )

        rows = cursor.fetchall()
        row_dict = {row[0]: (row[1], json.loads(row[2])) for row in rows}

        string_ids: List[Optional[str]] = []
        metadata: List[Optional[Dict[str, Any]]] = []
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

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
