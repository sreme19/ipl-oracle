# ipl-oracle

A multi-agent IPL XI optimizer. You give it a team code; it gives you back a
playing-eleven recommendation, a win probability, a toss call, and a written
brief explaining the reasoning.

There is no frontend, no web server, no cloud deployment. Everything runs
locally as a single CLI command.

## How it works

```
--team RCB
   │
   ▼
Fixture ─► Conditions ─► Scout(×2) ─► OpponentSelector ─► OpponentStrategist
                                                                      │
                                                                      ▼
                                                Strategist ─► Selector (MILP)
                                                                      │
                                                                      ▼
                                                  Simulator (Monte Carlo + MDP)
                                                                      │
                                                                      ▼
                                                            Narrator ──► text
```

Ten agents, each typed:

| Agent | Job |
| --- | --- |
| `FixtureAgent` | Looks up the team's next match (or honours overrides). |
| `ConditionsAgent` | Translates venue geometry + dew + pitch type into α/β formation weights. |
| `ScoutAgent` | Bayesian Normal-Normal posterior over each player's form. |
| `OpponentSelector` | Predictive MILP — what XI will the opponent field? |
| `OpponentStrategist` | Forecasts opponent score, phase strengths, builds the bipartite threat graph. |
| `StrategistAgent` | Meta-agent: picks deterministic / robust / Bayesian MILP based on uncertainty. |
| `SelectorAgent` | Prescriptive MILP — picks our XI to maximise matchup-aware objective. |
| `SimulatorAgent` | 10k Monte Carlo + MDP value iteration; emits win prob and toss recommendation. |
| MC Feedback Loop | Iteratively swaps marginal picks for bench alternatives and adopts any swap that lifts win probability by >0.5% (up to 5 rounds at 2k samples each). |
| `NarratorAgent` | Anthropic Claude — writes the natural-language brief from the trace. |
| `LinkedInAgent` | Deterministic templating — turns the trace into a shareable LinkedIn post. |

## Optimization techniques

- Mixed-integer linear programming (PuLP/CBC) — XI selection with role,
  overseas, and matchup constraints.
- Bertsimas–Sim Γ-robust counterpart — protects against opponent-XI
  uncertainty by taking the worst-case of the Γ most-uncertain players.
- Normal-Normal Bayesian form posterior + Thompson sampling — replaces
  ad-hoc EWM with principled exploration under uncertainty.
- Markov Decision Process value iteration — exact chase win-probability
  on a (overs, wickets, runs-needed) grid; complements Monte Carlo.
- Vectorised Monte Carlo — 10 000 ball-by-ball rollouts per innings,
  Wilson confidence interval.
- MC feedback loop — post-MILP swap search: tries replacing the 3
  weakest selected players with same-role bench alternatives (2k MC
  samples per candidate, δ=0.5% threshold, max 5 rounds) to close the
  gap between the MILP proxy objective and actual win probability.
- Platt scaling (`sklearn.linear_model.LogisticRegression`) — calibrates
  raw MC probabilities once you've logged real outcomes.
- Bipartite weighted matching — constructs threat edges between our
  batters and opponent bowlers.

## Technology

**Language & runtime**
- Python 3.10–3.12

**Core libraries**
- `pydantic` — typed contracts between agents
- `typer` + `rich` — CLI parsing and terminal rendering
- `numpy` / `scipy` — numerical core; vectorised Monte Carlo and Wilson CI
- `PuLP` (CBC backend) — mixed-integer linear programming solver
- `scikit-learn` — Platt scaling via `LogisticRegression`
- `anthropic` — Claude SDK for the Narrator agent

**Algorithms**
- Mixed-integer linear programming (PuLP/CBC) — prescriptive XI selection
- Bertsimas–Sim Γ-robust counterpart — opponent-XI uncertainty
- Normal-Normal Bayesian form posterior + Thompson sampling
- Markov Decision Process value iteration — exact chase win-probability grid
- Vectorised Monte Carlo with Wilson confidence interval
- Platt scaling — calibrates raw MC probabilities once outcomes are logged
- Bipartite weighted matching — batter × bowler threat graph

**Storage**
- SQLite (`~/.ipl-oracle/state.db`) — EWM form scores, calibration log, and full eval run history

**Quality & CI**
- `pytest` — covers MILP, optimisation primitives, orchestrator end-to-end, LinkedIn agent
- `ruff` — lint + import order
- GitHub Actions — runs `pytest` and `ruff check` on every push and PR

**Development environment**
- Built with [Claude Code](https://claude.com/claude-code) and [Windsurf](https://windsurf.com)
- macOS / Linux / WSL on Windows

## Install

```bash
git clone <repo>
cd ipl-oracle
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Interactive mode

Run `ipl-oracle` with no arguments for a guided prompt — the easiest way to get started:

```bash
ipl-oracle
```

You'll be asked to choose a mode, then provide team codes and a date:

```
─────────────────────── IPL ORACLE ───────────────────────

  What would you like to do?

   [1]  Pre-match strategy
        Recommend XI, toss & win probability for an upcoming fixture

   [2]  Post-match RCA
        Analyse why a result differed from the oracle's prediction

Mode [1/2]:
```

### Pre-match strategy (run)

```bash
# Most common: just the team — everything else inferred
ipl-oracle run --team RCB

# Override opponent / venue / date for what-ifs
ipl-oracle run --team RCB --opponent PBKS --venue "M Chinnaswamy Stadium" --match-date 2026-04-15

# Skip the LLM call (deterministic structured output only)
ipl-oracle run --team RCB --no-narrative

# Lock players in/out
ipl-oracle run --team RCB --must-include rcb_kohli --must-exclude rcb_maxwell

# Also emit a LinkedIn-style post
ipl-oracle run --team RCB --linkedin
```

Output is a structured terminal report with four sections: **Fixture → Verdict → Recommended XI → Strategy Brief**.

```
──────────────── IPL ORACLE — PRE-MATCH STRATEGY REPORT ────────────────

FIXTURE ─────────────────────────────────────────────────────────────────
  RCB vs GT  ·  Narendra Modi Stadium  ·  2026-04-30
  Protagonist: RCB  Opponent: GT

VERDICT ─────────────────────────────────────────────────────────────────
  Win Probability  82.9%  (95% CI 82.1% – 83.6%)
  Expected Runs    202    (198 – 206)
  Toss             BAT FIRST  (bat 82.9% vs bowl 66.7%)
  Mode             Bayesian

RECOMMENDED XI — RCB ───────────────────────────────────────────────────
  #   Name              Role             Selection Reason
  1   Virat Kohli       batter           top-of-order value despite matchup threat
  ...

STRATEGY BRIEF ─────────────────────────────────────────────────────────
  → Bat first. Set win-prob (82.9%) > chase win-prob (66.7%)...
```

### Post-match RCA (retro)

```bash
# Auto-fetch actual result from Cricsheet (no manual input needed)
ipl-oracle retro --team RCB --opponent GT \
    --venue "Narendra Modi Stadium" --match-date 2026-04-30 \
    --fetch-result

# Manual entry
ipl-oracle retro --team DC --opponent RCB \
    --venue "Arun Jaitley Stadium" --match-date 2026-04-27 \
    --actual-toss bowl --lost --runs-for 138 --runs-against 215
```

Output is a structured RCA report: **Match Summary → Prediction Accuracy → What Went Wrong → Key Takeaways**.
The calibration point `(predicted, actual)` is automatically logged to improve future Platt-scaled win probabilities.

### Other commands

```bash
# Inspect data
ipl-oracle fixtures --team RCB
ipl-oracle squad RCB
ipl-oracle enrichment-team RCB
ipl-oracle enrichment-match 2026-IPL-42

# Review past oracle runs
ipl-oracle history --team RCB --last 20

# Record match outcome against a prior run (feeds Platt calibration)
ipl-oracle record-outcome <run-id> --won
ipl-oracle record-outcome <run-id> --lost

# LinkedIn post (no LLM key required)
ipl-oracle linkedin --team RCB
```

## Refreshing data

The bundled dataset is a curated 2026 sample. To refresh from Cricsheet:

```bash
python scripts/ingest_cricsheet.py --output ipl_oracle/data --seasons 2024 2025
```

Future fixtures are not in Cricsheet — `ipl_oracle/data/fixtures/fixtures.json`
is hand-maintained from the IPL website and should be updated each season.

To regenerate analyst commentary enrichment and team markdown briefings:

```bash
python scripts/refresh_enrichment.py --season 2026
```

## State

`ipl-oracle` writes a SQLite file (`~/.ipl-oracle/state.db` by default) with three tables:

| Table | Contents |
| --- | --- |
| `form` | EWM-decayed form score and variance per player, updated each run |
| `calibration` | `(predicted, actual)` pairs used to fit the Platt scaler |
| `runs` | Full eval snapshot per run: strategy mode, win probability, CI width, toss decision, MC feedback rounds/delta, and whether an actual outcome was recorded |

Override the database path with the `IPL_ORACLE_STATE_DB` environment variable.

Each run prints its `run_id` at the bottom of the output. After the match is played, feed the result back with `ipl-oracle record-outcome <run-id> --won/--lost` — this writes to the calibration table so the Platt scaler improves on the next run.

## Tests

```bash
pytest -q
ruff check .
```
