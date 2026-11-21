"""SQLite persistence for reproducible experiment summaries."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DB = Path(__file__).resolve().parents[1] / "outputs" / "experiments.sqlite3"


class ExperimentDB:
    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    configuration TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    reconstruction_log TEXT NOT NULL
                )
            """)

    def _connect(self):
        return sqlite3.connect(self.path)

    def save(self, name, dataset, configuration, summary, reconstruction_log):
        if not name or not name.strip():
            raise ValueError("Experiment name cannot be empty.")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO experiments(name, created_at, dataset, configuration, summary, reconstruction_log) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    name.strip(), datetime.now(timezone.utc).isoformat(), dataset,
                    json.dumps(configuration, default=str), json.dumps(summary, default=str),
                    json.dumps(reconstruction_log, default=str),
                ),
            )

    def list(self):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, created_at, dataset, configuration, summary, reconstruction_log "
                "FROM experiments ORDER BY id DESC"
            ).fetchall()
        columns = ["id", "name", "created_at", "dataset", "configuration", "summary", "reconstruction_log"]
        return [dict(zip(columns, row)) for row in rows]
