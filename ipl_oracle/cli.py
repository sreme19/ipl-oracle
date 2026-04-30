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
from rich.text import Text

from .agents.linkedin import LinkedInAgent
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
    linkedin: bool = typer.Option(False, "--linkedin", help="Also emit a LinkedIn-style post"),
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
    if result.run_id:
        console.print(f"[dim]run logged → {result.run_id}[/dim]")

    if linkedin:
        post = LinkedInAgent().compose(result)
        console.print(Panel(post, title="LinkedIn post"))


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

    notes = []
    if result.own_xi.note:
        notes.append(f"[bold]{result.own_team} XI:[/bold] {result.own_xi.note}")
    if result.opponent_forecast.predicted_xi.note:
        notes.append(
            f"[bold]{result.opponent_team} XI:[/bold] {result.opponent_forecast.predicted_xi.note}"
        )
    if notes:
        console.print(Panel("\n".join(notes), title="[yellow]Constraint warnings[/yellow]"))

    if not no_narrative and result.narrative:
        console.print(Panel(result.narrative, title="Brief"))

    if json_out:
        sys.stdout.write(result.model_dump_json(indent=2))
        sys.stdout.write("\n")


@app.command()
def retro(
    team: str = typer.Option(..., "--team", "-t", help="Team code (e.g. DC, RCB)"),
    opponent: str | None = typer.Option(None, "--opponent", "-o"),
    venue: str | None = typer.Option(None, "--venue", "-v"),
    match_date: str | None = typer.Option(None, "--match-date", "-d"),
    actual_toss: str = typer.Option(..., "--actual-toss", help="What our team actually chose: bat | bowl"),
    won: bool = typer.Option(False, "--won/--lost", help="Did our team win?"),
    runs_for: int | None = typer.Option(None, "--runs-for", help="Runs we scored (optional)"),
    runs_against: int | None = typer.Option(None, "--runs-against", help="Runs we conceded (optional)"),
    sample_size: int = typer.Option(10000, "--sample-size"),
    log_calibration: bool = typer.Option(
        True, "--log-calibration/--no-log-calibration",
        help="Append (predicted, actual) to the calibration store",
    ),
    data_dir: str | None = typer.Option(None, "--data-dir"),
):
    """Compare what the oracle would have recommended against the actual result."""
    if actual_toss not in ("bat", "bowl"):
        console.print("[red]error:[/red] --actual-toss must be 'bat' or 'bowl'")
        raise typer.Exit(1)

    loader = DataLoader(data_dir)
    state = StateStore()
    orch = Orchestrator(loader=loader, state=state, narrator=None)
    request = OracleRequest(
        team=team,
        opponent=opponent,
        venue=venue,
        match_date=_parse_date(match_date),
        sample_size=sample_size,
    )
    try:
        result = orch.run(request)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    fx = result.fixture
    wp = result.win_probability
    toss = result.toss
    actual_wp = 1.0 if won else 0.0
    delta_pp = (wp.win_probability - actual_wp) * 100.0

    console.print(
        Panel(
            f"[bold]{fx.home_team}[/bold] vs [bold]{fx.away_team}[/bold]  "
            f"@ {fx.venue}  ({fx.match_date.isoformat()})\n"
            f"Perspective: [bold]{result.own_team}[/bold]",
            title="Retrospective",
        )
    )

    table = Table(title="Predicted vs Actual")
    table.add_column("Metric")
    table.add_column("Predicted")
    table.add_column("Actual")
    table.add_column("Delta")
    table.add_row(
        "Outcome",
        f"win-prob {wp.win_probability:.1%}",
        "Won" if won else "Lost",
        f"{delta_pp:+.1f} pp"
        if won == (wp.win_probability >= 0.5)
        else f"[red]{delta_pp:+.1f} pp[/red]",
    )
    if runs_for is not None:
        if actual_toss == "bat":
            pred = wp.expected_runs
        else:
            pred = result.opponent_forecast.expected_score
        table.add_row(
            "Runs (us)",
            f"{pred:.0f}",
            str(runs_for),
            f"{runs_for - pred:+.0f}",
        )
    if runs_against is not None:
        if actual_toss == "bat":
            pred = result.opponent_forecast.expected_score
        else:
            pred = wp.expected_runs
        table.add_row(
            "Runs (them)",
            f"{pred:.0f}",
            str(runs_against),
            f"{runs_against - pred:+.0f}",
        )
    table.add_row(
        "Toss",
        toss.decision,
        actual_toss,
        "[green]matched[/green]" if toss.decision == actual_toss else "[red]mismatched[/red]",
    )
    console.print(table)

    diagnostics: list[str] = []
    if toss.decision != actual_toss:
        gap = abs(toss.win_prob_batting_first - toss.win_prob_bowling_first) * 100.0
        diagnostics.append(
            f"⚠ Toss mismatch — model preferred '{toss.decision}' "
            f"(set {toss.win_prob_batting_first:.1%} vs chase {toss.win_prob_bowling_first:.1%}, "
            f"gap {gap:.1f} pp). Choosing the other side likely cost win probability."
        )
    if won and wp.win_probability < 0.5:
        diagnostics.append(
            f"✓ Upset win — model gave only {wp.win_probability:.1%}. "
            "Calibration store will pull future predictions toward this outcome."
        )
    elif not won and wp.win_probability > 0.5:
        diagnostics.append(
            f"✗ Model overestimated — predicted {wp.win_probability:.1%} but lost. "
            f"Inspect: opponent score forecast was {result.opponent_forecast.expected_score:.0f}, "
            f"par {result.conditions.par_score:.0f}."
        )
    if runs_against is not None:
        opp_pred = result.opponent_forecast.expected_score
        if runs_against > opp_pred + 25:
            diagnostics.append(
                f"✗ Conceded {runs_against - opp_pred:+.0f} above forecast — phase strategy miss. "
                f"Weakest opponent batting phase was '{min(result.opponent_forecast.batting_phase_strengths, key=result.opponent_forecast.batting_phase_strengths.get)}'; "
                "if you bowled medium-pace there instead of attacking, that's the leak."
            )
    if result.own_xi.note:
        diagnostics.append(f"⚠ Own XI constraint warning: {result.own_xi.note}")
    if result.opponent_forecast.predicted_xi.note:
        diagnostics.append(
            f"⚠ Opponent XI constraint warning: {result.opponent_forecast.predicted_xi.note}"
        )

    if diagnostics:
        console.print(Panel("\n\n".join(diagnostics), title="Diagnostics"))

    if log_calibration:
        state.add_calibration_point(wp.win_probability, actual_wp)
        console.print(
            f"[dim]Calibration point logged: predicted={wp.win_probability:.3f}, "
            f"actual={actual_wp:.0f}[/dim]"
        )


@app.command()
def linkedin(
    team: str = typer.Option(..., "--team", "-t", help="Team code (e.g. RCB, PBKS, MI)"),
    opponent: str | None = typer.Option(None, "--opponent", "-o"),
    venue: str | None = typer.Option(None, "--venue", "-v"),
    match_date: str | None = typer.Option(None, "--match-date", "-d"),
    sample_size: int = typer.Option(10000, "--sample-size"),
    repo_url: str | None = typer.Option(None, "--repo-url", help="Override the repo link in the post"),
    data_dir: str | None = typer.Option(None, "--data-dir"),
):
    """Run the pipeline and emit a LinkedIn-style post to stdout."""
    loader = DataLoader(data_dir)
    state = StateStore()
    orch = Orchestrator(loader=loader, state=state, narrator=None)
    request = OracleRequest(
        team=team,
        opponent=opponent,
        venue=venue,
        match_date=_parse_date(match_date),
        sample_size=sample_size,
    )
    try:
        result = orch.run(request)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    agent = LinkedInAgent(repo_url=repo_url) if repo_url else LinkedInAgent()
    sys.stdout.write(agent.compose(result))
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


@app.command(name="enrichment-team")
def enrichment_team(
    team: str = typer.Argument(..., help="Team code"),
    season: int | None = typer.Option(None, "--season", help="Season year"),
    markdown: bool = typer.Option(False, "--markdown", help="Print generated markdown briefing"),
    data_dir: str | None = typer.Option(None, "--data-dir"),
):
    """Show synthesized analyst enrichment for one team."""
    loader = DataLoader(data_dir)
    try:
        insight = loader.load_team_insight(team, season=season)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if markdown:
        try:
            briefing = loader.load_team_briefing_markdown(team, season=season)
            sys.stdout.write(briefing)
            if not briefing.endswith("\n"):
                sys.stdout.write("\n")
            return
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(1) from exc

    table = Table(title=f"Analyst Insight — {team.upper()}")
    table.add_column("Type", style="cyan")
    table.add_column("Details")
    table.add_row("Take", insight.get("analyst_take", ""))
    table.add_row("Strengths", "; ".join(insight.get("strengths", [])))
    table.add_row("Risks", "; ".join(insight.get("risks", [])))
    table.add_row("Style", ", ".join(insight.get("style_tags", [])))
    console.print(table)


@app.command(name="enrichment-match")
def enrichment_match(
    match_id: str = typer.Argument(..., help="Match ID, e.g. 2026-IPL-42"),
    season: int | None = typer.Option(None, "--season", help="Season year"),
    data_dir: str | None = typer.Option(None, "--data-dir"),
):
    """Show synthesized analyst commentary for one fixture."""
    loader = DataLoader(data_dir)
    try:
        fx = loader.find_fixture_commentary(match_id, season=season)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    panel_text = (
        f"{fx['home_team']} vs {fx['away_team']} @ {fx['venue']} ({fx['match_date']})\n\n"
        f"{fx['insight_commentary']}\n\n"
        "Watch for:\n"
        f"- {fx['watch_for'][0]}\n"
        f"- {fx['watch_for'][1]}\n"
        f"- {fx['watch_for'][2]}"
    )
    console.print(Panel(panel_text, title=f"Enrichment {match_id}"))


@app.command()
def history(
    team: str | None = typer.Option(None, "--team", "-t", help="Filter by team code"),
    last: int = typer.Option(20, "--last", "-n", help="Number of runs to show"),
):
    """Review past oracle runs logged in state.db."""
    state = StateStore()
    rows = state.run_history(team=team, limit=last)
    if not rows:
        console.print("[yellow]No runs logged yet.[/yellow]")
        raise typer.Exit()

    table = Table(title="Run History", show_lines=False)
    table.add_column("Date", style="dim")
    table.add_column("Team")
    table.add_column("vs")
    table.add_column("Venue")
    table.add_column("Mode")
    table.add_column("Win %", justify="right")
    table.add_column("CI width", justify="right")
    table.add_column("Toss")
    table.add_column("Refine", justify="right")
    table.add_column("Outcome", justify="center")
    table.add_column("Run ID", style="dim")

    for r in rows:
        ts = datetime.fromtimestamp(r["run_at"]).strftime("%Y-%m-%d %H:%M")
        win_pct = f"{r['win_probability']:.1%}"
        ci = f"{r['ci_width']:.3f}"
        outcome = "[green]✓[/green]" if r["outcome_recorded"] else "[dim]—[/dim]"
        mode_colour = {"robust": "yellow", "bayesian": "cyan", "deterministic": "white"}.get(
            r["strategy_mode"], "white"
        )
        table.add_row(
            ts,
            r["team"],
            r["opponent"],
            r["venue"][:28],
            Text(r["strategy_mode"], style=mode_colour),
            win_pct,
            ci,
            r["toss_decision"],
            str(r["refinement_rounds"]),
            outcome,
            r["run_id"][:8] + "…",
        )

    console.print(table)
    console.print(
        f"\n[dim]Full run IDs shown truncated. "
        f"DB path: {StateStore().path}[/dim]"
    )


@app.command(name="record-outcome")
def record_outcome(
    run_id: str = typer.Argument(..., help="Run ID from 'ipl-oracle history'"),
    won: bool = typer.Option(..., "--won/--lost", help="Did the team win?"),
):
    """Record the actual match result for a past run (feeds Platt calibration)."""
    state = StateStore()
    ok = state.record_outcome(run_id, won)
    if not ok:
        console.print(f"[red]Run ID not found:[/red] {run_id}")
        raise typer.Exit(1)
    result_str = "[green]WIN[/green]" if won else "[red]LOSS[/red]"
    console.print(f"Outcome recorded: {result_str} for run {run_id[:8]}…")
    console.print("[dim]Platt scaler will use this on the next run.[/dim]")


if __name__ == "__main__":
    app()
