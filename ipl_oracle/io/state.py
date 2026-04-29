"""Local SQLite store for form decay (EWM), calibration log, and eval run history."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas import OracleResult

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
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                run_at REAL NOT NULL,
                team TEXT NOT NULL,
                opponent TEXT NOT NULL,
                venue TEXT NOT NULL,
                strategy_mode TEXT NOT NULL,
                gamma REAL NOT NULL,
                selected_xi TEXT NOT NULL,
                win_probability REAL NOT NULL,
                ci_lower REAL NOT NULL,
                ci_upper REAL NOT NULL,
                ci_width REAL NOT NULL,
                toss_decision TEXT NOT NULL,
                refinement_rounds INTEGER NOT NULL,
                solve_time_ms REAL NOT NULL,
                sample_size INTEGER NOT NULL,
                calibrated INTEGER NOT NULL,
                outcome_recorded INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ form

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

    # ----------------------------------------------------------- calibration

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

    # --------------------------------------------------------------- runs / evals

    def log_run(self, result: OracleResult) -> str:
        """Persist a completed oracle run. Returns the run_id."""
        run_id = result.run_id or str(uuid.uuid4())
        wp = result.win_probability
        ci_lower, ci_upper = wp.confidence_interval
        strategy = result.decision_trace.get("strategy", {})
        self._conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, run_at, team, opponent, venue,
                strategy_mode, gamma, selected_xi,
                win_probability, ci_lower, ci_upper, ci_width,
                toss_decision, refinement_rounds,
                solve_time_ms, sample_size, calibrated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                time.time(),
                result.own_team,
                result.opponent_team,
                result.fixture.venue,
                result.own_xi.mode.value,
                float(strategy.get("gamma", 0.0)),
                json.dumps([p.player_id for p in result.own_xi.selected_xi]),
                wp.win_probability,
                ci_lower,
                ci_upper,
                ci_upper - ci_lower,
                result.toss.decision,
                int(strategy.get("refinement_attempts", 0)),
                result.own_xi.solve_time_ms,
                wp.sample_size,
                int(wp.calibrated),
            ),
        )
        self._conn.commit()
        return run_id

    def run_history(
        self,
        team: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Return recent runs as dicts, newest first."""
        if team:
            rows = self._conn.execute(
                """
                SELECT run_id, run_at, team, opponent, venue,
                       strategy_mode, gamma, win_probability,
                       ci_lower, ci_upper, ci_width, toss_decision,
                       refinement_rounds, solve_time_ms, sample_size,
                       calibrated, outcome_recorded
                FROM runs
                WHERE team = ?
                ORDER BY run_at DESC LIMIT ?
                """,
                (team.upper(), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT run_id, run_at, team, opponent, venue,
                       strategy_mode, gamma, win_probability,
                       ci_lower, ci_upper, ci_width, toss_decision,
                       refinement_rounds, solve_time_ms, sample_size,
                       calibrated, outcome_recorded
                FROM runs
                ORDER BY run_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        keys = [
            "run_id", "run_at", "team", "opponent", "venue",
            "strategy_mode", "gamma", "win_probability",
            "ci_lower", "ci_upper", "ci_width", "toss_decision",
            "refinement_rounds", "solve_time_ms", "sample_size",
            "calibrated", "outcome_recorded",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def record_outcome(self, run_id: str, won: bool) -> bool:
        """Record actual match result for a past run. Feeds the Platt calibration log.
        Returns False if run_id not found."""
        row = self._conn.execute(
            "SELECT win_probability FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return False
        predicted = row[0]
        actual = 1.0 if won else 0.0
        self.add_calibration_point(predicted, actual)
        self._conn.execute(
            "UPDATE runs SET outcome_recorded = 1 WHERE run_id = ?", (run_id,)
        )
        self._conn.commit()
        return True

    def close(self) -> None:
        self._conn.close()
