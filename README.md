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

Nine agents, each typed:

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
| `NarratorAgent` | Anthropic Claude — writes the natural-language brief from the trace. |

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
- Platt scaling (`sklearn.linear_model.LogisticRegression`) — calibrates
  raw MC probabilities once you've logged real outcomes.
- Bipartite max-weight matching (NetworkX) — constructs threat edges
  between our batters and opponent bowlers.

## Install

```bash
git clone <repo>
cd ipl-oracle
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Most common: just the team — everything else inferred
ipl-oracle run --team RCB

# Override opponent / venue / date for what-ifs
ipl-oracle run --team RCB --opponent PBKS --venue "M Chinnaswamy Stadium" --match-date 2026-04-15

# Skip the LLM call (deterministic structured output only)
ipl-oracle run --team RCB --no-narrative

# Lock players in/out
ipl-oracle run --team RCB --must-include rcb_kohli --must-exclude rcb_maxwell

# Inspect data
ipl-oracle fixtures --team RCB
ipl-oracle squad RCB
```

Sample output (truncated):

```
Fixture: RCB vs PBKS @ M Chinnaswamy Stadium (2026-04-15)

Predicted PBKS XI:
 1. Shreyas Iyer (batsman)
 2. Prabhsimran Singh (wicket_keeper)
 ...

Recommended XI:
 1. Virat Kohli      (batsman)        top-of-order value despite matchup threat (4.2)
 2. Phil Salt        (wicket_keeper)  wk slot — meets role minimum
 ...

win prob: 56.3% (95% CI 52.1% – 60.4%)
toss: bat — set win-prob 56.3% ≥ chase win-prob 53.1%; par 195 suggests posting first
mode: deterministic
```

## Refreshing data

The bundled dataset is a curated 2026 sample. To refresh from Cricsheet:

```bash
python scripts/ingest_cricsheet.py --output ipl_oracle/data --seasons 2024 2025
```

Future fixtures are not in Cricsheet — `ipl_oracle/data/fixtures/fixtures.json`
is hand-maintained from the IPL website and should be updated each season.

## State

`ipl-oracle` writes a small SQLite file (`~/.ipl-oracle/state.db` by default)
to track exponentially-weighted form scores and a calibration log of
predicted-vs-actual win probabilities. Override with the
`IPL_ORACLE_STATE_DB` environment variable.

## Tests

```bash
pytest -q
ruff check .
```
