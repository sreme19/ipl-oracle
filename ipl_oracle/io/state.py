"""Local SQLite store for form decay (EWM) and calibration log."""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

_DEFAULT_DB = Path.home() / ".ipl-oracle" / "state.db"


class StateStore:
    def __init__(self, path: str | os.PathLike | None = None):
        env = os.environ.get("IPL_ORACLE_STATE_DB")
        self.path = Path(path or env or _DEFAULT_DB).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS form (
                player_id TEXT PRIMARY KEY,
                form_score REAL NOT NULL,
                form_variance REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted REAL NOT NULL,
                actual REAL NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    def get_form(self, player_id: str) -> tuple[float, float] | None:
        row = self._conn.execute(
            "SELECT form_score, form_variance FROM form WHERE player_id = ?",
            (player_id,),
        ).fetchone()
        return (row[0], row[1]) if row else None

    def update_form_ewm(
        self, player_id: str, observation: float, alpha: float = 0.4
    ) -> tuple[float, float]:
        """Apply EWM update: f_t = α·obs + (1-α)·f_{t-1}; variance via EWMVar."""
        prior = self.get_form(player_id)
        if prior is None:
            new_score, new_var = observation, 0.1
        else:
            prev_score, prev_var = prior
            new_score = alpha * observation + (1 - alpha) * prev_score
            diff = observation - prev_score
            new_var = (1 - alpha) * (prev_var + alpha * diff * diff)
        self._conn.execute(
            "INSERT OR REPLACE INTO form (player_id, form_score, form_variance, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (player_id, new_score, new_var, time.time()),
        )
        self._conn.commit()
        return new_score, new_var

    def add_calibration_point(self, predicted: float, actual: float) -> None:
        self._conn.execute(
            "INSERT INTO calibration (predicted, actual, created_at) VALUES (?, ?, ?)",
            (predicted, actual, time.time()),
        )
        self._conn.commit()

    def calibration_history(self, limit: int = 100) -> list[tuple[float, float]]:
        rows = self._conn.execute(
            "SELECT predicted, actual FROM calibration ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def close(self) -> None:
        self._conn.close()
