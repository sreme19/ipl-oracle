"""Typer-based CLI for ipl-oracle."""
from __future__ import annotations

import json
import sys
from datetime import date as Date
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .agents.linkedin import LinkedInAgent
from .agents.narrator import NarratorAgent
from .io import CricinfoFetchError, DataLoader, StateStore, fetch_match_result
from .orchestrator import Orchestrator
from .schemas import OracleRequest

app = typer.Typer(
    help="Multi-agent IPL XI optimizer.",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()

_TEAM_CODES = ["RCB", "GT", "MI", "CSK", "KKR", "DC", "PBKS", "RR", "SRH", "LSG"]


# ---------------------------------------------------------------------------
# Default entrypoint — interactive prompt when no subcommand is given
# ---------------------------------------------------------------------------

@app.callback()
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _interactive()


def _interactive() -> None:
    console.print()
    console.print(Rule("[bold cyan]IPL ORACLE[/bold cyan]", style="cyan"))
    console.print()
    console.print("  [bold]What would you like to do?[/bold]\n")
    console.print("   [cyan][1][/cyan]  [bold]Pre-match strategy[/bold]")
    console.print("        Recommend XI, toss & win probability for an upcoming fixture\n")
    console.print("   [cyan][2][/cyan]  [bold]Post-match RCA[/bold]")
    console.print("        Analyse why a result differed from the oracle's prediction\n")

    mode = typer.prompt("Mode", prompt_suffix=" [1/2]: ").strip()
    while mode not in ("1", "2"):
        console.print("[red]Enter 1 or 2.[/red]")
        mode = typer.prompt("Mode", prompt_suffix=" [1/2]: ").strip()

    console.print()
    console.print(f"  Valid team codes: [dim]{', '.join(_TEAM_CODES)}[/dim]")
    team = typer.prompt("  Protagonist team code").strip().upper()
    opponent = typer.prompt("  Opponent team code").strip().upper()
    match_date = typer.prompt("  Match date (YYYY-MM-DD)").strip()

    console.print()

    if mode == "1":
        _run_interactive(team, opponent, match_date)
    else:
        _retro_interactive(team, opponent, match_date)


def _run_interactive(team: str, opponent: str, match_date: str) -> None:
    loader = DataLoader(None)
    state = StateStore()
    narrator = NarratorAgent()
    orch = Orchestrator(loader=loader, state=state, narrator=narrator)
    request = OracleRequest(
        team=team,
        opponent=opponent,
        match_date=_parse_date(match_date),
    )
    try:
        result = orch.run(request)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    _render(result, json_out=False, no_narrative=False)
    if result.run_id:
        console.print(f"[dim]run logged → {result.run_id}[/dim]")


def _retro_interactive(team: str, opponent: str, match_date: str) -> None:
    console.print("  [bold]Fetch actual result automatically from Cricsheet?[/bold]")
    auto = typer.confirm("  Auto-fetch", default=True)

    parsed_date = _parse_date(match_date)
    actual_toss: str | None = None
    runs_for: int | None = None
    runs_against: int | None = None
    won = False

    if auto and parsed_date:
        console.print(f"[dim]  Fetching {team} vs {opponent} on {parsed_date} from Cricsheet…[/dim]")
        try:
            live = fetch_match_result(team, opponent, parsed_date)
            actual_toss = live.toss
            runs_for = live.runs_for
            runs_against = live.runs_against
            won = live.won
            console.print(
                f"  [green]Fetched:[/green] toss={actual_toss}, "
                f"runs_for={runs_for}, runs_against={runs_against}, "
                f"{'won' if won else 'lost'}\n"
            )
        except CricinfoFetchError as exc:
            console.print(f"  [yellow]Auto-fetch failed:[/yellow] {exc}")
            console.print("  Falling back to manual entry.\n")

    if actual_toss is None:
        actual_toss = typer.prompt("  Actual toss (bat/bowl)").strip().lower()
        won = typer.confirm("  Did the team win?", default=False)
        rf = typer.prompt("  Runs scored (leave blank to skip)", default="").strip()
        ra = typer.prompt("  Runs conceded (leave blank to skip)", default="").strip()
        runs_for = int(rf) if rf.isdigit() else None
        runs_against = int(ra) if ra.isdigit() else None

    loader = DataLoader(None)
    state = StateStore()
    orch = Orchestrator(loader=loader, state=state, narrator=None)
    request = OracleRequest(
        team=team, opponent=opponent, match_date=parsed_date
    )
    try:
        result = orch.run(request)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    _render_retro(result, actual_toss, won, runs_for, runs_against, log_cal=True, state=state)


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

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
    linkedin_post: bool = typer.Option(False, "--linkedin", help="Also emit a LinkedIn-style post"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory"),
):
    """Run the multi-agent pipeline and print a structured strategy report."""
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

    if linkedin_post:
        post = LinkedInAgent().compose(result)
        console.print(Panel(post, title="LinkedIn post"))


# ---------------------------------------------------------------------------
# Structured pre-match report renderer
# ---------------------------------------------------------------------------

def _render(result, json_out: bool, no_narrative: bool) -> None:
    fx = result.fixture
    wp = result.win_probability
    toss = result.toss

    # ── Header ──────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]IPL ORACLE — PRE-MATCH STRATEGY REPORT[/bold]", style="cyan"))

    # ── Fixture ─────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("FIXTURE", style="dim", align="left"))
    console.print(
        f"  [bold]{fx.home_team}[/bold] vs [bold]{fx.away_team}[/bold]"
        f"  ·  {fx.venue}  ·  {fx.match_date.isoformat()}"
    )
    console.print(f"  Protagonist: [bold cyan]{result.own_team}[/bold cyan]  "
                  f"Opponent: [bold]{result.opponent_team}[/bold]")

    # ── Verdict ─────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("VERDICT", style="dim", align="left"))
    toss_label = "BAT FIRST" if toss.decision == "bat" else "BOWL FIRST"
    toss_style = "green" if toss.decision == "bat" else "yellow"
    console.print(
        f"  Win Probability  [bold]{wp.win_probability:.1%}[/bold]  "
        f"[dim](95% CI {wp.confidence_interval[0]:.1%} – {wp.confidence_interval[1]:.1%})[/dim]"
    )
    console.print(
        f"  Expected Runs    [bold]{wp.expected_runs:.0f}[/bold]  "
        f"[dim]({wp.expected_runs_ci[0]:.0f} – {wp.expected_runs_ci[1]:.0f})[/dim]"
    )
    console.print(
        f"  Toss             [{toss_style}][bold]{toss_label}[/bold][/{toss_style}]  "
        f"[dim](bat {toss.win_prob_batting_first:.1%} vs bowl {toss.win_prob_bowling_first:.1%})[/dim]"
    )
    console.print(f"  Mode             [dim]{result.own_xi.mode.value}[/dim]")
    console.print()
    console.print(f"  [italic]{toss.rationale}[/italic]")

    # ── Recommended XI ──────────────────────────────────────────────────────
    console.print()
    console.print(Rule(f"RECOMMENDED XI — {result.own_team}", style="dim", align="left"))
    xi_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    xi_table.add_column("#", justify="right", style="dim")
    xi_table.add_column("Name", min_width=22)
    xi_table.add_column("Role", min_width=18)
    xi_table.add_column("Selection Reason")
    for i, p in enumerate(result.own_xi.selected_xi, 1):
        reason = result.own_xi.slot_reasons.get(p.player_id, "—")
        xi_table.add_row(str(i), p.name, p.role.value, reason)
    console.print(xi_table)

    # ── Opponent XI ─────────────────────────────────────────────────────────
    console.print()
    console.print(Rule(f"OPPONENT XI — {result.opponent_team} (PREDICTED)", style="dim", align="left"))
    opp_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    opp_table.add_column("#", justify="right", style="dim")
    opp_table.add_column("Name", min_width=22)
    opp_table.add_column("Role")
    for i, p in enumerate(result.opponent_forecast.predicted_xi.selected_xi, 1):
        opp_table.add_row(str(i), p.name, p.role.value)
    console.print(opp_table)

    # ── Strategy Brief ──────────────────────────────────────────────────────
    if not no_narrative and result.narrative:
        console.print()
        console.print(Rule("STRATEGY BRIEF", style="dim", align="left"))
        for line in result.narrative.strip().splitlines():
            console.print(f"  {line}")

    # ── Constraint Warnings ─────────────────────────────────────────────────
    warnings = []
    if result.own_xi.note:
        warnings.append(f"[bold]{result.own_team} XI:[/bold] {result.own_xi.note}")
    if result.opponent_forecast.predicted_xi.note:
        warnings.append(
            f"[bold]{result.opponent_team} XI:[/bold] {result.opponent_forecast.predicted_xi.note}"
        )
    if warnings:
        console.print()
        console.print(Rule("[yellow]CONSTRAINT WARNINGS[/yellow]", style="yellow dim", align="left"))
        for w in warnings:
            console.print(f"  ⚠ {w}")

    console.print()
    console.print(Rule(style="cyan"))

    if json_out:
        sys.stdout.write(result.model_dump_json(indent=2))
        sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# retro command
# ---------------------------------------------------------------------------

@app.command()
def retro(
    team: str = typer.Option(..., "--team", "-t", help="Team code (e.g. DC, RCB)"),
    opponent: str | None = typer.Option(None, "--opponent", "-o"),
    venue: str | None = typer.Option(None, "--venue", "-v"),
    match_date: str | None = typer.Option(None, "--match-date", "-d"),
    actual_toss: str | None = typer.Option(None, "--actual-toss", help="What our team actually chose: bat | bowl"),
    won: bool = typer.Option(False, "--won/--lost", help="Did our team win?"),
    runs_for: int | None = typer.Option(None, "--runs-for", help="Runs we scored (optional)"),
    runs_against: int | None = typer.Option(None, "--runs-against", help="Runs we conceded (optional)"),
    fetch_result: bool = typer.Option(
        False, "--fetch-result",
        help="Auto-fetch toss, scores, and outcome from Cricsheet (requires --opponent and --match-date)",
    ),
    sample_size: int = typer.Option(10000, "--sample-size"),
    log_calibration: bool = typer.Option(
        True, "--log-calibration/--no-log-calibration",
        help="Append (predicted, actual) to the calibration store",
    ),
    data_dir: str | None = typer.Option(None, "--data-dir"),
):
    """Root cause analysis — compare oracle predictions against the actual result."""
    if fetch_result:
        if not opponent or not match_date:
            console.print("[red]error:[/red] --fetch-result requires --opponent and --match-date")
            raise typer.Exit(1)
        parsed_date = _parse_date(match_date)
        if parsed_date is None:
            console.print("[red]error:[/red] --match-date must be in YYYY-MM-DD format")
            raise typer.Exit(1)
        console.print(f"[dim]Fetching result for {team} vs {opponent} on {parsed_date} from Cricsheet…[/dim]")
        try:
            live = fetch_match_result(team, opponent, parsed_date)
        except CricinfoFetchError as exc:
            console.print(f"[red]Cricsheet fetch failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        actual_toss = live.toss
        runs_for = live.runs_for
        runs_against = live.runs_against
        won = live.won
        console.print(
            f"[green]Fetched:[/green] {live.match_title} — "
            f"toss={actual_toss}, runs_for={runs_for}, runs_against={runs_against}, "
            f"{'won' if won else 'lost'}"
        )

    if actual_toss is None:
        console.print("[red]error:[/red] --actual-toss is required (or use --fetch-result)")
        raise typer.Exit(1)
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

    _render_retro(result, actual_toss, won, runs_for, runs_against, log_cal=log_calibration, state=state)


# ---------------------------------------------------------------------------
# Structured RCA renderer
# ---------------------------------------------------------------------------

def _render_retro(result, actual_toss: str, won: bool, runs_for: int | None,
                  runs_against: int | None, log_cal: bool, state: StateStore) -> None:
    fx = result.fixture
    wp = result.win_probability
    toss = result.toss
    actual_wp = 1.0 if won else 0.0
    delta_pp = (wp.win_probability - actual_wp) * 100.0

    # ── Header ──────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]IPL ORACLE — ROOT CAUSE ANALYSIS[/bold]", style="red"))

    # ── Match Summary ────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("MATCH SUMMARY", style="dim", align="left"))
    result_label = "[green]WON[/green]" if won else "[red]LOST[/red]"
    console.print(
        f"  [bold]{fx.home_team}[/bold] vs [bold]{fx.away_team}[/bold]"
        f"  ·  {fx.venue}  ·  {fx.match_date.isoformat()}"
    )
    console.print(
        f"  Perspective: [bold cyan]{result.own_team}[/bold cyan]  ·  "
        f"Result: {result_label}  ·  Toss: {actual_toss}"
    )

    # ── Prediction Accuracy ─────────────────────────────────────────────────
    console.print()
    console.print(Rule("PREDICTION ACCURACY", style="dim", align="left"))
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Metric", min_width=14)
    table.add_column("Predicted", justify="right", min_width=16)
    table.add_column("Actual", justify="right", min_width=10)
    table.add_column("Delta", justify="right")

    delta_style = "green" if won == (wp.win_probability >= 0.5) else "red"
    table.add_row(
        "Outcome",
        f"win {wp.win_probability:.1%}",
        "Won" if won else "Lost",
        Text(f"{delta_pp:+.1f} pp", style=delta_style),
    )
    if runs_for is not None:
        pred_us = wp.expected_runs if actual_toss == "bat" else result.opponent_forecast.expected_score
        d = runs_for - pred_us
        table.add_row(
            "Runs (us)",
            f"{pred_us:.0f}",
            str(runs_for),
            Text(f"{d:+.0f}", style="green" if d >= 0 else "red"),
        )
    if runs_against is not None:
        pred_them = result.opponent_forecast.expected_score if actual_toss == "bat" else wp.expected_runs
        d = runs_against - pred_them
        table.add_row(
            "Runs (them)",
            f"{pred_them:.0f}",
            str(runs_against),
            Text(f"{d:+.0f}", style="red" if d > 0 else "green"),
        )
    toss_match = toss.decision == actual_toss
    table.add_row(
        "Toss call",
        toss.decision,
        actual_toss,
        Text("✓ matched", style="green") if toss_match else Text("✗ mismatched", style="red"),
    )
    console.print(table)

    # ── What Went Wrong ──────────────────────────────────────────────────────
    diagnostics: list[tuple[str, str]] = []  # (symbol+style, text)

    if not toss_match:
        gap = abs(toss.win_prob_batting_first - toss.win_prob_bowling_first) * 100.0
        diagnostics.append((
            "[yellow]⚠[/yellow]",
            f"[bold]Toss decision mismatch[/bold] — model preferred '{toss.decision}' "
            f"(bat {toss.win_prob_batting_first:.1%} vs bowl {toss.win_prob_bowling_first:.1%}, "
            f"gap {gap:.1f} pp). Going the other way likely surrendered win probability."
        ))

    if won and wp.win_probability < 0.5:
        diagnostics.append((
            "[green]✓[/green]",
            f"[bold]Upset win[/bold] — model gave only {wp.win_probability:.1%}. "
            "Calibration store will pull future estimates toward this outcome."
        ))
    elif not won and wp.win_probability > 0.5:
        diagnostics.append((
            "[red]✗[/red]",
            f"[bold]Model overestimated win probability[/bold] — predicted {wp.win_probability:.1%} but lost. "
            f"Opponent score forecast was {result.opponent_forecast.expected_score:.0f} "
            f"(par {result.conditions.par_score:.0f}). "
            "The run model and/or matchup edges were too optimistic."
        ))

    if runs_for is not None:
        pred_us = wp.expected_runs if actual_toss == "bat" else result.opponent_forecast.expected_score
        miss = runs_for - pred_us
        if abs(miss) > 20:
            diagnostics.append((
                "[red]✗[/red]" if miss < 0 else "[yellow]⚠[/yellow]",
                f"[bold]Batting model missed by {miss:+.0f} runs[/bold] — "
                f"predicted {pred_us:.0f}, scored {runs_for}. "
                + ("Batting underperformed the model's phase-by-phase projection." if miss < 0
                   else "Batting overperformed — model was conservative.")
            ))

    if runs_against is not None:
        opp_pred = result.opponent_forecast.expected_score
        over = runs_against - opp_pred
        if over > 25:
            weakest_phase = min(
                result.opponent_forecast.batting_phase_strengths,
                key=result.opponent_forecast.batting_phase_strengths.get,
            )
            diagnostics.append((
                "[red]✗[/red]",
                f"[bold]Conceded {over:+.0f} above forecast[/bold] — bowling phase strategy leak. "
                f"Opponent's weakest phase was '{weakest_phase}'; "
                "aggressive pace bowling there instead of containing may have changed the outcome."
            ))

    if result.own_xi.note:
        diagnostics.append(("[yellow]⚠[/yellow]", f"[bold]XI constraint warning:[/bold] {result.own_xi.note}"))
    if result.opponent_forecast.predicted_xi.note:
        diagnostics.append((
            "[yellow]⚠[/yellow]",
            f"[bold]Opponent XI constraint warning:[/bold] {result.opponent_forecast.predicted_xi.note}"
        ))

    if diagnostics:
        console.print()
        console.print(Rule("WHAT WENT WRONG", style="dim", align="left"))
        for symbol, text in diagnostics:
            console.print(f"  {symbol}  {text}")
            console.print()

    # ── Key Takeaways ────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("KEY TAKEAWAYS", style="dim", align="left"))

    # Largest error source
    errors: list[tuple[float, str]] = []
    if runs_for is not None:
        pred_us = wp.expected_runs if actual_toss == "bat" else result.opponent_forecast.expected_score
        errors.append((abs(runs_for - pred_us), f"batting model ({runs_for - pred_us:+.0f} runs)"))
    if runs_against is not None:
        pred_them = result.opponent_forecast.expected_score if actual_toss == "bat" else wp.expected_runs
        errors.append((abs(runs_against - pred_them), f"bowling model ({runs_against - pred_them:+.0f} runs)"))
    if errors:
        errors.sort(reverse=True)
        console.print(f"  • Largest error source: [bold]{errors[0][1]}[/bold]")

    if not toss_match:
        console.print(f"  • Toss call diverged from reality — review pitch/dew model inputs")

    console.print(
        f"  • Prediction: [bold]{wp.win_probability:.1%}[/bold]  →  "
        f"Actual: {'[green]WIN[/green]' if won else '[red]LOSS[/red]'}"
    )

    if log_cal:
        state.add_calibration_point(wp.win_probability, actual_wp)
        console.print(
            f"\n  [dim]Calibration point logged: predicted={wp.win_probability:.3f}, "
            f"actual={actual_wp:.0f}[/dim]"
        )

    console.print()
    console.print(Rule(style="red"))


# ---------------------------------------------------------------------------
# Remaining commands (unchanged)
# ---------------------------------------------------------------------------

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
    console.print(f"\n[dim]DB path: {StateStore().path}[/dim]")


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


def _parse_date(s: str | None) -> Date | None:
    if not s:
        return None
    return datetime.fromisoformat(s).date()


if __name__ == "__main__":
    app()
