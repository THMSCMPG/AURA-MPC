"""pi/logger.py – Local logging: SQLite database + JSONL flat file.

Every SensorPacket is persisted locally before being forwarded to the
orchestrator, giving a durable audit trail and enabling the ``replay.py``
script to re-send historical data.

SQLite schema
-------------
    CREATE TABLE packets (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        sequence    INTEGER,
        type_id     INTEGER,
        raw_json    TEXT    NOT NULL
    );

JSONL file
----------
    One JSON object per line, appended to ``~/edge-aura/logs/packets.jsonl``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_DB_PATH   = Path.home() / "edge-aura" / "logs" / "packets.db"
_DEFAULT_JSONL_PATH = Path.home() / "edge-aura" / "logs" / "packets.jsonl"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS packets (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    sequence  INTEGER,
    raw_json  TEXT    NOT NULL
);
"""

_INSERT_SQL = "INSERT INTO packets (timestamp, sequence, raw_json) VALUES (?, ?, ?)"


class Logger:
    """Dual-sink logger: SQLite + JSONL.

    Parameters
    ----------
    db_path : Path | str
        Path to the SQLite database file.
    jsonl_path : Path | str
        Path to the JSONL log file.
    """

    def __init__(
        self,
        db_path:    Path | str = _DEFAULT_DB_PATH,
        jsonl_path: Path | str = _DEFAULT_JSONL_PATH,
    ) -> None:
        self._db_path    = Path(db_path)
        self._jsonl_path = Path(jsonl_path)
        self._conn: sqlite3.Connection | None = None
        self._jsonl_fh = None

    # ------------------------------------------------------------------
    def open(self) -> None:
        """Open the database and log file, creating them if necessary."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()
        self._jsonl_fh = open(self._jsonl_path, "a", encoding="utf-8")  # noqa: WPS515
        log.info("Logger opened: db=%s  jsonl=%s", self._db_path, self._jsonl_path)

    def close(self) -> None:
        """Flush and close both sinks."""
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._jsonl_fh:
            self._jsonl_fh.close()
            self._jsonl_fh = None

    def __enter__(self) -> "Logger":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    def write(self, packet: dict[str, Any]) -> None:
        """Persist a SensorPacket dict to both sinks.

        Parameters
        ----------
        packet : dict
            A SensorPacket as returned by :meth:`PacketBuilder.build`.
        """
        ts  = packet.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()
        seq = packet.get("sequence")
        raw = json.dumps(packet)

        if self._conn is not None:
            try:
                self._conn.execute(_INSERT_SQL, (ts, seq, raw))
                self._conn.commit()
            except sqlite3.Error as exc:
                log.error("SQLite write error: %s", exc)

        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.write(raw + "\n")
                self._jsonl_fh.flush()
            except OSError as exc:
                log.error("JSONL write error: %s", exc)
