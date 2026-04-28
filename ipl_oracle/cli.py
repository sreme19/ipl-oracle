"""Typer-based CLI for ipl-oracle."""
from __future__ import annotations

import json
import sys
from datetime import date as Date
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agents.narrator import NarratorAgent
from .io import DataLoader, StateStore
from .orchestrator import Orchestrator
from .schemas import OracleRequest

app = typer.Typer(help="Multi-agent IPL XI optimizer.", add_completion=False)
console = Console()


def _parse_date(s: str | None) -> Date | None:
    if not s:
        return None
    return datetime.fromisoformat(s).date()


@app.command()
def run(
    team: str = typer.Option(..., "--team", "-t", help="Team code (e.g. RCB, PBKS, MI)"),
    opponent: str | None = typer.Option(None, "--opponent", "-o", help="Override opponent code"),
    venue: str | None = typer.Option(None, "--venue", "-v", help="Override venue name"),
    match_date: str | None = typer.Option(None, "--match-date", "-d", help="ISO date (YYYY-MM-DD)"),
    formation_bias: str = typer.Option("balanced", "--formation", help="batting | balanced | bowling"),
    must_include: list[str] = typer.Option([], "--must-include", help="Player IDs to lock in"),
    must_exclude: list[str] = typer.Option([], "--must-exclude", help="Player IDs to lock out"),
    sample_size: int = typer.Option(10000, "--sample-size", help="Monte Carlo rollouts"),
    json_out: bool = typer.Option(False, "--json", help="Emit full JSON trace alongside narrative"),
    no_narrative: bool = typer.Option(False, "--no-narrative", help="Skip the Anthropic call"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory"),
):
    """Run the multi-agent pipeline and print a natural-language verdict."""
    loader = DataLoader(data_dir)
    state = StateStore()
    narrator = None if no_narrative else NarratorAgent()
    orch = Orchestrator(loader=loader, state=state, narrator=narrator)
    request = OracleRequest(
        team=team,
        opponent=opponent,
        venue=venue,
        match_date=_parse_date(match_date),
        formation_bias=formation_bias,
        must_include=list(must_include),
        must_exclude=list(must_exclude),
        sample_size=sample_size,
    )

    try:
        result = orch.run(request)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    _render(result, json_out=json_out, no_narrative=no_narrative)


def _render(result, json_out: bool, no_narrative: bool) -> None:
    fx = result.fixture
    console.print(
        Panel(
            f"[bold]{fx.home_team}[/bold] vs [bold]{fx.away_team}[/bold]  "
            f"@ {fx.venue}  ({fx.match_date.isoformat()})",
            title="Fixture",
        )
    )

    table = Table(title=f"Predicted XI — {result.opponent_team}")
    table.add_column("#", justify="right")
    table.add_column("Name")
    table.add_column("Role")
    for i, p in enumerate(result.opponent_forecast.predicted_xi.selected_xi, 1):
        table.add_row(str(i), p.name, p.role.value)
    console.print(table)

    table = Table(title=f"Recommended XI — {result.own_team}")
    table.add_column("#", justify="right")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Reason")
    for i, p in enumerate(result.own_xi.selected_xi, 1):
        reason = result.own_xi.slot_reasons.get(p.player_id, "")
        table.add_row(str(i), p.name, p.role.value, reason)
    console.print(table)

    wp = result.win_probability
    console.print(
        Panel(
            f"win prob: [bold]{wp.win_probability:.1%}[/bold] "
            f"(95% CI {wp.confidence_interval[0]:.1%} – {wp.confidence_interval[1]:.1%})\n"
            f"expected runs: {wp.expected_runs:.0f}  "
            f"({wp.expected_runs_ci[0]:.0f} – {wp.expected_runs_ci[1]:.0f})\n"
            f"toss: [bold]{result.toss.decision}[/bold] — {result.toss.rationale}\n"
            f"mode: {result.own_xi.mode.value}",
            title="Verdict",
        )
    )

    if not no_narrative and result.narrative:
        console.print(Panel(result.narrative, title="Brief"))

    if json_out:
        sys.stdout.write(result.model_dump_json(indent=2))
        sys.stdout.write("\n")


@app.command()
def fixtures(
    team: str | None = typer.Option(None, "--team", "-t"),
    data_dir: str | None = typer.Option(None, "--data-dir"),
):
    """List upcoming fixtures (optionally filtered by team)."""
    loader = DataLoader(data_dir)
    fxs = loader.load_fixtures()
    if team:
        team_u = team.upper()
        fxs = [f for f in fxs if team_u in (f.home_team, f.away_team)]
    fxs.sort(key=lambda f: f.match_date)
    table = Table(title="Fixtures")
    table.add_column("Date")
    table.add_column("Home")
    table.add_column("Away")
    table.add_column("Venue")
    for f in fxs:
        table.add_row(f.match_date.isoformat(), f.home_team, f.away_team, f.venue)
    console.print(table)


@app.command()
def squad(
    team: str = typer.Argument(..., help="Team code"),
    data_dir: str | None = typer.Option(None, "--data-dir"),
):
    """Inspect a squad as JSON."""
    loader = DataLoader(data_dir)
    sq = loader.load_squad(team)
    sys.stdout.write(json.dumps(sq.model_dump(), indent=2, default=str))
    sys.stdout.write("\n")


if __name__ == "__main__":
    app()
