"""Smoke test for `ipl-oracle retro`."""
from __future__ import annotations

import os

from typer.testing import CliRunner

from ipl_oracle.cli import app

runner = CliRunner()


def test_retro_runs_and_emits_diagnostics(tmp_path):
    os.environ["IPL_ORACLE_STATE_DB"] = str(tmp_path / "state.db")
    result = runner.invoke(
        app,
        [
            "retro",
            "--team", "DC",
            "--opponent", "RCB",
            "--venue", "Arun Jaitley Stadium",
            "--match-date", "2026-04-27",
            "--actual-toss", "bowl",
            "--lost",
            "--runs-against", "210",
            "--runs-for", "150",
            "--sample-size", "500",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Retrospective" in result.output
    assert "Predicted vs Actual" in result.output
    # Toss mismatch diagnostic should fire (model says bat, actual was bowl).
    assert "Toss mismatch" in result.output
    # Model overestimated: predicted >50% but lost.
    assert "Model overestimated" in result.output or "overestimated" in result.output


def test_retro_invalid_toss_value(tmp_path):
    os.environ["IPL_ORACLE_STATE_DB"] = str(tmp_path / "state.db")
    result = runner.invoke(
        app,
        [
            "retro",
            "--team", "DC",
            "--opponent", "RCB",
            "--venue", "Arun Jaitley Stadium",
            "--match-date", "2026-04-27",
            "--actual-toss", "spin",
            "--won",
        ],
    )
    assert result.exit_code != 0
    assert "must be 'bat' or 'bowl'" in result.output
